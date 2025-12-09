import sys
import os
import re
import subprocess
import time
import statistics

def run_calibration(args):
    """
    Runs resnet_search.py with calibration arguments, captures profiling output,
    and returns average times for Main and Worker.
    """
    # 1. Construct command: Add -v and --max_chunks 5 (enough to stabilize)
    cmd = [sys.executable, 'resnet_search.py'] + args
    
    # Ensure verbose is on
    if '-v' not in cmd and '--verbose' not in cmd:
        cmd.append('-v')
        
    # Ensure limit is set (unless user set it)
    if '--max_chunks' not in cmd:
        cmd.extend(['--max_chunks', '5'])
        
    print(f"[*] Starting calibration run: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    
    worker_times = []
    main_times = []
    
    # Regex
    re_worker = re.compile(r"PROFILE \[Worker\]: .*? = ([\d\.]+) s/chunk")
    re_main = re.compile(r"PROFILE \[Main\]: .*? = ([\d\.]+) s/chunk")
    
    try:
        start_time = time.time()
        while True:
            # Timeout 300s
            if time.time() - start_time > 300:
                print("[!] Timeout waiting for data.")
                break
                
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if not line:
                continue
                
            # Echo output gently
            if "PROFILE" in line:
                print(f"    {line.strip()}")
            elif "Processing" in line or "file:" in line:
                print(f"    {line.strip()}")
            
            # Parse
            mw = re_worker.search(line)
            if mw:
                val = float(mw.group(1))
                if len(worker_times) > 0 or val < 100: 
                    worker_times.append(val)
            
            mm = re_main.search(line)
            if mm:
                val = float(mm.group(1))
                if len(main_times) > 0 or val < 100:
                    main_times.append(val)
                    
            # Stop if finished naturally (subprocess ends)
            
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                process.kill()
    
    if not worker_times or not main_times:
        print("[!] Failed to capture sufficient profiling data.")
        return None, None
        
    # Drop first sample (warmup) if we have enough
    avg_w = statistics.mean(worker_times[1:]) if len(worker_times) > 1 else worker_times[0]
    avg_m = statistics.mean(main_times[1:]) if len(main_times) > 1 else main_times[0]
    
    return avg_w, avg_m

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python auto_tune_threads.py [arguments for resnet_search.py]")
        print("Example: python auto_tune_threads.py -i ../ -o ../output -re *.fits")
        return

    print("=== Auto-Tuning ResNet Search Threads (Non-Destructive) ===")
    
    # Default settings assumed for the initial run
    cur_m = 0.4
    cur_w = 0.5
    
    # Check if user passed ratios already
    if '--onnx_ratio' in args:
        idx = args.index('--onnx_ratio')
        cur_m = float(args[idx+1])
    if '--worker_ratio' in args:
        idx = args.index('--worker_ratio')
        cur_w = float(args[idx+1])

    print(f"[*] Base Configuration: Main(ONNX)={cur_m*100:.1f}%, Worker={cur_w*100:.1f}%")
    
    # Run
    avg_w_time, avg_m_time = run_calibration(args)
    if not avg_w_time or not avg_m_time:
        print("[!] Calibration failed.")
        return

    print(f"\n[*] Calibration Results (Avg):")
    print(f"    Worker Time : {avg_w_time:.4f} s")
    print(f"    Main Time   : {avg_m_time:.4f} s")
    
    if avg_w_time == 0 or avg_m_time == 0:
        print("[!] Invalid zero time captured.")
        return

    # Calculate Workload units
    # Workload = Time * Ratio (Core Share)
    workload_main = avg_m_time * cur_m
    workload_worker = avg_w_time * cur_w
    total_workload = workload_main + workload_worker
    
    # Optimal Allocation: Ratio_new = Total_Allocatable * (Workload / Total_Workload)
    # Reserve 10% for system overhead
    total_allocatable = 0.90 
    
    new_m = total_allocatable * (workload_main / total_workload)
    new_w = total_allocatable * (workload_worker / total_workload)
    
    # Clamp bounds (min 0.1, max 0.8)
    new_m = max(0.1, min(0.8, new_m))
    new_w = max(0.1, min(0.8, new_w))
    
    print(f"\n[*] Optimization Analysis:")
    print(f"    Workload Balance: Main={workload_main:.2f} units vs Worker={workload_worker:.2f} units")
    print(f"    Optimal Split   : Main={new_m*100:.1f}% / Worker={new_w*100:.1f}%")
    
    print("\n[+] RECOMMENDED COMMAND:")
    print("-" * 60)
    # Reconstruct command with new ratios
    base_cmd = "python resnet_search.py " + " ".join([a for a in args if a not in ['--onnx_ratio', '--worker_ratio', '-v', '--verbose']])
    # Strip existing ratio values if present in args list logic (simplified here)
    
    print(f"{base_cmd} --onnx_ratio {new_m:.2f} --worker_ratio {new_w:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
