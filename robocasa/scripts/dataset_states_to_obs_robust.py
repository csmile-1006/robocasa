"""
Robust script to extract observations from low-dimensional simulation states in a robocasa dataset.
This version includes improved multiprocessing with better error handling, timeouts, and deadlock prevention.
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback
import signal
from contextlib import contextmanager
from typing import List

import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout handling."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


class RobustMultiprocessingManager:
    """Manages multiprocessing with robust error handling and adaptive timeouts."""
    
    def __init__(self, num_processes: int, total_work_items: int = 0):
        self.num_processes = num_processes
        self.total_work_items = total_work_items
        self.processes = []
        self.queues = {}
        self.shared_vars = {}
        self.lock = multiprocessing.Lock()
        self.shutdown_event = multiprocessing.Event()
        
        # Adaptive timeout settings
        self.base_timeout_per_item = 60  # Base timeout per work item in seconds
        self.min_timeout = 60  # Minimum total timeout
        self.max_timeout = 3600  # Maximum total timeout (1 hour)
        self.heartbeat_timeout = 300  # Timeout for heartbeat (5 minutes)
        
        # Progress tracking
        self.last_progress_time = multiprocessing.Value('d', time.time())
        self.writer_processed = multiprocessing.Value('i', 0)  # Writer's actual progress
        
    def setup_shared_resources(self, num_demos: int):
        """Set up shared multiprocessing resources."""
        self.total_work_items = num_demos
        self.queues['work'] = multiprocessing.Queue(maxsize=num_demos)
        self.queues['results'] = multiprocessing.Queue(maxsize=num_demos * 2)
        self.queues['errors'] = multiprocessing.Queue()
        self.queues['heartbeat'] = multiprocessing.Queue()  # For progress tracking
        
        self.shared_vars['total_samples'] = multiprocessing.Value('i', 0)
        self.shared_vars['num_finished'] = multiprocessing.Value('i', 0)
        self.shared_vars['current_work'] = multiprocessing.Array('i', self.num_processes)
        
    def calculate_adaptive_timeout(self) -> int:
        """Calculate adaptive timeout based on work remaining."""
        with self.lock:
            remaining_items = max(0, self.total_work_items - self.writer_processed.value)
            if remaining_items == 0:
                return self.min_timeout
            
            # Calculate timeout based on remaining work
            estimated_timeout = remaining_items * self.base_timeout_per_item
            adaptive_timeout = max(self.min_timeout, min(estimated_timeout, self.max_timeout))
            
            print(f"Adaptive timeout: {adaptive_timeout}s (remaining items: {remaining_items})")
            return adaptive_timeout
    
    
    def check_heartbeat(self) -> bool:
        """Check if processes are making progress."""
        with self.lock:
            time_since_progress = time.time() - self.last_progress_time.value
            return time_since_progress < self.heartbeat_timeout
        
    def cleanup_processes(self):
        """Clean up all processes and resources."""
        print("Cleaning up processes...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Terminate processes that don't respond to shutdown
        for process in self.processes:
            if process.is_alive():
                print(f"Terminating process {process.name}")
                process.terminate()
                
        # Wait for processes to finish with timeout
        for process in self.processes:
            if process.is_alive():
                try:
                    with timeout(5):
                        process.join()
                except TimeoutError:
                    print(f"Force killing process {process.name}")
                    process.kill()
                    process.join()
                    
        # Clear process list
        self.processes.clear()
        
        # Close queues
        for queue_name, queue_obj in self.queues.items():
            try:
                while not queue_obj.empty():
                    queue_obj.get_nowait()
                queue_obj.close()
                queue_obj.join_thread()
            except Exception:
                pass


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )
    traj_len = states.shape[0]
    
    for t in range(traj_len):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)

    # convert list of dict to dict of list for obs dictionaries
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def safe_queue_put(queue_obj, item, timeout_seconds=30):
    """Safely put an item in a queue with timeout."""
    try:
        with timeout(timeout_seconds):
            queue_obj.put(item, block=True, timeout=timeout_seconds)
        return True
    except (queue.Full, TimeoutError):
        print(f"Warning: Failed to put item in queue after {timeout_seconds}s timeout")
        return False


def safe_queue_get(queue_obj, timeout_seconds=30):
    """Safely get an item from a queue with timeout."""
    try:
        with timeout(timeout_seconds):
            return queue_obj.get(block=True, timeout=timeout_seconds)
    except (queue.Empty, TimeoutError):
        return None


def worker_process(
    process_id: int,
    args,
    manager: RobustMultiprocessingManager,
    demos: List[str]
):
    """Worker process for extracting trajectories."""
    env = None
    f = None
    
    try:
        print(f"Worker {process_id} starting...")
        
        # Create environment
        if args.add_datagen_info:
            import mimicgen.utils.file_utils as MG_FileUtils
            env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        else:
            env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
            
        if args.generative_textures:
            env_meta["env_kwargs"]["generative_textures"] = "100p"
        if args.randomize_cameras:
            env_meta["env_kwargs"]["randomize_cameras"] = True
            
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            reward_shaping=args.shaped,
        )
        
        # Open dataset file
        f = h5py.File(args.dataset, "r")
        
        processed_count = 0
        
        while not manager.shutdown_event.is_set():
            # Get work item with longer timeout - processes should wait for work
            work_item = safe_queue_get(manager.queues['work'], timeout_seconds=30)
            if work_item is None:
                if manager.shutdown_event.is_set():
                    break
                # Send heartbeat to indicate we're still alive but waiting
                safe_queue_put(manager.queues['heartbeat'], [process_id, "waiting", time.time()])
                continue
                
            demo_idx = work_item
            ep = demos[demo_idx]
            
            try:
                # Process the episode
                states = f["data/{}/states".format(ep)][()]
                initial_state = dict(states=states[0])
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
                initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
                
                actions = f["data/{}/actions".format(ep)][()]
                
                traj = extract_trajectory(
                    env=env,
                    initial_state=initial_state,
                    states=states,
                    actions=actions,
                    done_mode=args.done_mode,
                    add_datagen_info=args.add_datagen_info,
                )
                
                # Copy rewards/dones if requested
                if args.copy_rewards:
                    traj["rewards"] = f["data/{}/rewards".format(ep)][()]
                if args.copy_dones:
                    traj["dones"] = f["data/{}/dones".format(ep)][()]
                
                # Put result in queue
                result_item = [ep, traj, process_id, demo_idx]
                if not safe_queue_put(manager.queues['results'], result_item):
                    print(f"Worker {process_id}: Failed to put result for episode {ep}")
                    continue
                
                # Send heartbeat
                safe_queue_put(manager.queues['heartbeat'], [process_id, "completed", time.time(), ep])
                    
                processed_count += 1
                print(f"Worker {process_id}: Processed episode {ep} ({processed_count} total)")
                
            except Exception as e:
                error_msg = f"Worker {process_id} error processing episode {ep}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                
                # Put error in error queue
                safe_queue_put(manager.queues['errors'], [process_id, ep, str(e)])
                
                # Recreate environment on error
                if env is not None:
                    del env
                    env = EnvUtils.create_env_for_data_processing(
                        env_meta=env_meta,
                        camera_names=args.camera_names,
                        camera_height=args.camera_height,
                        camera_width=args.camera_width,
                        reward_shaping=args.shaped,
                    )
        
        print(f"Worker {process_id} finished, processed {processed_count} episodes")
        
    except Exception as e:
        error_msg = f"Worker {process_id} fatal error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        safe_queue_put(manager.queues['errors'], [process_id, "FATAL", str(e)])
        
    finally:
        # Cleanup
        if f is not None:
            f.close()
        if env is not None:
            del env
            
        # Signal completion
        with manager.lock:
            manager.shared_vars['num_finished'].value += 1


def writer_process(
    args,
    output_path: str,
    manager: RobustMultiprocessingManager,
    num_demos: int
):
    """Writer process for saving trajectories to file."""
    f_out = None
    f_in = None
    
    try:
        print("Writer process starting...")
        
        f_in = h5py.File(args.dataset, "r")
        f_out = h5py.File(output_path, "w")
        data_grp = f_out.create_group("data")
        
        start_time = time.time()
        num_processed = 0
        processed_episodes = set()
        
        while num_processed < num_demos and not manager.shutdown_event.is_set():
            # Get result with timeout
            result_item = safe_queue_get(manager.queues['results'], timeout_seconds=10)
            if result_item is None:
                # Check if all workers are done
                if manager.shared_vars['num_finished'].value >= manager.num_processes:
                    break
                continue
                
            ep, traj, process_id, demo_idx = result_item
            
            # Skip if already processed
            if ep in processed_episodes:
                continue
            processed_episodes.add(ep)
            
            try:
                # Write trajectory to file
                ep_data_grp = data_grp.create_group(ep)
                ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                
                # Write observations
                for k in traj["obs"]:
                    if args.no_compress:
                        ep_data_grp.create_dataset(
                            "obs/{}".format(k), data=np.array(traj["obs"][k])
                        )
                    else:
                        ep_data_grp.create_dataset(
                            "obs/{}".format(k),
                            data=np.array(traj["obs"][k]),
                            compression="gzip",
                        )
                
                # Write datagen info if present
                if "datagen_info" in traj:
                    for k in traj["datagen_info"]:
                        ep_data_grp.create_dataset(
                            "datagen_info/{}".format(k),
                            data=np.array(traj["datagen_info"][k]),
                        )
                
                # Copy action dict if applicable
                if "data/{}/action_dict".format(ep) in f_in:
                    action_dict = f_in["data/{}/action_dict".format(ep)]
                    for k in action_dict:
                        ep_data_grp.create_dataset(
                            "action_dict/{}".format(k),
                            data=np.array(action_dict[k][()]),
                        )
                
                # Episode metadata
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
                ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
                
                # Update shared variables
                with manager.lock:
                    manager.shared_vars['total_samples'].value += traj["actions"].shape[0]
                    manager.writer_processed.value += 1  # Update writer progress
                    manager.last_progress_time.value = time.time()  # Update heartbeat
                
                num_processed += 1
                elapsed = time.time() - start_time
                rate = elapsed / num_processed if num_processed > 0 else 0
                
                print(f"Writer: Processed {num_processed}/{num_demos} episodes "
                      f"(episode {ep} from worker {process_id}), "
                      f"rate: {rate:.2f} sec/demo")
                
            except Exception as e:
                print(f"Writer error processing episode {ep}: {str(e)}")
                print(traceback.format_exc())
                continue
        
        # Copy mask if present
        if "mask" in f_in:
            f_in.copy("mask", f_out)
        
        # Write global metadata
        data_grp.attrs["total"] = manager.shared_vars['total_samples'].value
        
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.generative_textures:
            env_meta["env_kwargs"]["generative_textures"] = "100p"
        if args.randomize_cameras:
            env_meta["env_kwargs"]["randomize_cameras"] = True
            
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            reward_shaping=args.shaped,
        )
        
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        print(f"Writer: Wrote {manager.shared_vars['total_samples'].value} total samples to {output_path}")
        
    except Exception as e:
        print(f"Writer process fatal error: {str(e)}")
        print(traceback.format_exc())
        
    finally:
        if f_out is not None:
            f_out.close()
        if f_in is not None:
            f_in.close()


def dataset_states_to_obs_robust(args):
    """Main function with robust multiprocessing."""
    # Setup output path
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    # Get demo list
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    print(f"Processing {num_demos} demonstrations with {args.num_procs} processes")

    # Setup multiprocessing manager
    manager = RobustMultiprocessingManager(args.num_procs, total_work_items=num_demos)
    manager.setup_shared_resources(num_demos)
    
    # Override timeout settings if provided
    if hasattr(args, 'timeout_per_item'):
        manager.base_timeout_per_item = args.timeout_per_item
    if hasattr(args, 'max_timeout'):
        manager.max_timeout = args.max_timeout
    if hasattr(args, 'heartbeat_timeout'):
        manager.heartbeat_timeout = args.heartbeat_timeout
    
    try:
        # Add work items to queue
        for i in range(num_demos):
            safe_queue_put(manager.queues['work'], i)
        
        # Start worker processes
        for i in range(args.num_procs):
            process = multiprocessing.Process(
                target=worker_process,
                args=(i, args, manager, demos),
                name=f"Worker-{i}"
            )
            manager.processes.append(process)
            process.start()
        
        # Start writer process
        writer_process_obj = multiprocessing.Process(
            target=writer_process,
            args=(args, output_path, manager, num_demos),
            name="Writer"
        )
        manager.processes.append(writer_process_obj)
        writer_process_obj.start()
        
        # Wait for completion with adaptive timeout and heartbeat monitoring
        print("Waiting for processes to complete...")
        start_time = time.time()
        last_heartbeat_check = time.time()
        last_progress_report = time.time()
        
        # Main loop: wait for all worker processes to finish or for shutdown
        while (manager.shared_vars['num_finished'].value < args.num_procs and 
               not manager.shutdown_event.is_set()):
            time.sleep(5)  # Check every 5 seconds instead of every second

            current_time = time.time()

            # If all episodes have been processed, break out of the loop
            with manager.lock:
                processed = manager.writer_processed.value
            if processed >= num_demos:
                print(f"All {num_demos} episodes processed. Exiting progress loop.")
                break

            # Check heartbeats every 30 seconds
            if current_time - last_heartbeat_check > 30:
                if not manager.check_heartbeat():
                    print(f"No progress for {manager.heartbeat_timeout}s, checking for stuck processes...")
                    # Check if any processes are actually stuck
                    alive_processes = [p for p in manager.processes if p.is_alive()]
                    if len(alive_processes) > 0:
                        print(f"Found {len(alive_processes)} alive processes, continuing...")
                        manager.last_progress_time.value = time.time()  # Reset heartbeat
                    else:
                        print("No alive processes found, shutting down...")
                        manager.shutdown_event.set()
                        break
                last_heartbeat_check = current_time

            # Report progress every 60 seconds
            if current_time - last_progress_report > 60:
                with manager.lock:
                    processed = manager.writer_processed.value
                    remaining = max(0, num_demos - processed)
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0

                    print(f"Progress: {processed}/{num_demos} episodes processed "
                          f"({processed/num_demos*100:.1f}%), "
                          f"rate: {rate:.2f} eps/sec, ETA: {eta/60:.1f} min")
                last_progress_report = current_time
            
            # # Check adaptive timeout
            # adaptive_timeout = manager.calculate_adaptive_timeout()
            # if current_time - start_time > adaptive_timeout:
            #     print(f"Adaptive timeout reached ({adaptive_timeout}s), shutting down...")
            #     manager.shutdown_event.set()
            #     break
        
        # Wait for writer to finish
        if writer_process_obj.is_alive():
            writer_process_obj.join(timeout=30)
            if writer_process_obj.is_alive():
                print("Writer process did not finish gracefully, terminating...")
                writer_process_obj.terminate()
                writer_process_obj.join()
        
        print("All processes completed")
        
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
        manager.shutdown_event.set()
        
    except Exception as e:
        print(f"Fatal error in main process: {str(e)}")
        print(traceback.format_exc())
        manager.shutdown_event.set()
        
    finally:
        # Cleanup
        manager.cleanup_processes()
        
        # Post-process the output file
        if os.path.exists(output_path):
            try:
                DatasetUtils.extract_action_dict(dataset=output_path)
                DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
                
                # Create filtered datasets
                for num_demos_filter in [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 
                                       125, 150, 200, 250, 300, 400, 500, 600, 700, 
                                       800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 
                                       5000, 10000]:
                    DatasetUtils.filter_dataset_size(
                        output_path,
                        num_demos=num_demos_filter,
                    )
                print("Post-processing completed")
            except Exception as e:
                print(f"Error in post-processing: {str(e)}")

    print("Robust multiprocessing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )
    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--randomize_cameras", action="store_true")
    
    parser.add_argument(
        "--timeout_per_item",
        type=int,
        default=30,
        help="Base timeout per work item in seconds (default: 30)",
    )
    
    parser.add_argument(
        "--max_timeout",
        type=int,
        default=3600,
        help="Maximum total timeout in seconds (default: 3600 = 1 hour)",
    )
    
    parser.add_argument(
        "--heartbeat_timeout",
        type=int,
        default=300,
        help="Heartbeat timeout in seconds (default: 300 = 5 minutes)",
    )

    args = parser.parse_args()
    dataset_states_to_obs_robust(args)
