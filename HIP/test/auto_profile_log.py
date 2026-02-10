import os
import shutil
import subprocess

def run_workflow():
    required_helpers = ['Makefile', 'bp.py', 'gen_mcin']
    items = [f for f in os.scandir('.') if f.is_dir()]
    
    for item in items:
        folder_name = item.name
        
        if folder_name.endswith('_profile'):
            continue
            
        main_hip_path = os.path.join(folder_name, 'main.hip')
        if os.path.exists(main_hip_path):
            profile_folder = f"{folder_name}_profile"
            print(f"--- Processing {folder_name} -> Logging to {profile_folder}/profile.log ---")
            
            if os.path.exists(profile_folder):
                shutil.rmtree(profile_folder)
            
            shutil.copytree(folder_name, profile_folder)
            
            for helper in required_helpers:
                if os.path.exists(helper):
                    shutil.copy2(helper, profile_folder)

            make_commands = [
                ["make", "profile"],
                ["make", "BP_LV=l1", "profile_bp"],
                ["make", "BP_LV=l2", "profile_bp"],
                ["make", "BP_LV=all", "profile_bp"]
            ]

            original_dir = os.getcwd()
            try:
                os.chdir(profile_folder)
                # Open the log file in the current profile directory
                with open("profile.log", "a") as log_file:
                    for cmd in make_commands:
                        log_file.write(f"\n--- Executing: {' '.join(cmd)} ---\n")
                        log_file.flush() # Ensure header is written immediately
                        
                        # Redirect both stdout and stderr to the log file
                        result = subprocess.run(
                            cmd, 
                            stdout=log_file, 
                            stderr=log_file, 
                            text=True
                        )
                        
                        if result.returncode != 0:
                            log_file.write(f"ERROR: Command failed with return code {result.returncode}\n")
            except Exception as e:
                print(f"An error occurred while processing {profile_folder}: {e}")
            finally:
                os.chdir(original_dir)

    print("\n--- Workflow completed ---")

if __name__ == "__main__":
    run_workflow()
