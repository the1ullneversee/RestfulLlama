# Welcome to the RestfulLlama Project!

This project is designed to be run via Docker with GPU support. While we’ve made every effort to ensure a smooth experience, different hardware configurations, especially CUDA drivers, may cause issues. This project has been tested on Bath’s Hex Cloud, but some parts have only been tested on private cloud options due to the unavailability of high VRAM GPUs for MSc students.

## Requirements

- **High VRAM GPUs**: Ideally, use servers with GPUs that have around 40GB of VRAM.
- **Pre-flight Checks**: The project includes pre-flight checks to inform you of what you should be able to run.

## Getting Started

1. **Clone the Repository**: Log on to your server and clone the repository using the following command:
    
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/the1ullneversee/RestfulLlama.git
    ```
    
2. **Hex Cloud Users**: If you are using the Hex cloud, reserve the storage space and clone the repository in a location with faster storage, such as `/mnt/fast0` or `/mnt/faster0`. Reserve the area with this command (adjust the path as needed):
    
    ```bash
    hare reserve /mnt/faster0/tk698/project/RestfulLlama/
    ```
    
3. **Run the Project**: Make the `project_runner.sh` script executable and run it:
    
    ```bash
    chmod +x project_runner.sh
    ./project_runner.sh
    ```
    
    This process will build the Docker container and run it, attaching itself when done. Please note that this may take some time.
    

## Using the Application

Once inside the application, you can select various options. The CLI usage is covered in the attached project video. Be aware that operations like data generation can take a significant amount of time due to the nature of synthetic dataset generation.

## Browsing the Code

You can browse the code by attaching an IDE. To do this, perform an SSH loopback:

1. Find the container IP:
    
    ```bash
    hare inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' restful_llama
    ```
    
2. Use the IP in the following command in a new terminal:
    
    ```bash
    ssh -L 6000:172.17.0.3:22 -J tk698@aching.cs.bath.ac.uk
    ```
    A password will be required, this is contained with the Dockerfile.

## Additional Information

For more detailed information, please visit the Hex Cloud documentation.

For any queries, feel free to contact: tk698@bath.ac.uk