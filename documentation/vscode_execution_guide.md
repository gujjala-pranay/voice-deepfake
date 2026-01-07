# VS Code Execution Guide: Audio Deepfake Detection

This guide provides the exact steps to set up and run the project within **Visual Studio Code**.

## 1. Prerequisites
- **VS Code** installed.
- **Python Extension** for VS Code installed.
- **Python 3.11+** installed on your machine.

## 2. Setup Process

1.  **Open Project in VS Code**:
    - Launch VS Code.
    - Go to `File > Open Folder...` and select the `audio-deepfake-detection` folder.

2.  **Create Virtual Environment**:
    - Open the integrated terminal in VS Code (`Ctrl+` ` or `Terminal > New Terminal`).
    - Run the following commands:
      ```bash
      python -m venv venv
      ```
    - VS Code will likely ask if you want to select this environment for the workspace. Click **Yes**.
    - If not, click on the Python version in the bottom-right corner and select the interpreter inside the `venv` folder.

3.  **Install Dependencies**:
    - In the same terminal, run:
      ```bash
      pip install -r requirements.txt
      ```

## 3. Running the Project

We have provided pre-configured launch settings in the `.vscode` folder.

### Option A: Using the Run and Debug Side Bar (Recommended)
1.  Click on the **Run and Debug** icon on the left sidebar (or press `Ctrl+Shift+D`).
2.  In the dropdown menu at the top, you will see three options:
    - **Python: FastAPI**: Starts the backend API on port 8000.
    - **Python: Streamlit**: Starts the web interface on port 8501.
    - **Python: Train Model**: Starts the training script.
3.  Select the desired configuration and press the **Green Play Button** (F5).

### Option B: Manual Execution via Terminal
If you prefer the terminal, use these commands:
- **Start API**: `uvicorn main:app --reload --port 8000`
- **Start UI**: `streamlit run app.py --server.port 8501`

## 4. Dataset Handling (Medium Size)
The project is configured to handle a **medium-sized dataset** (~10,000 samples) by default. You can adjust this in `utils.py` by modifying the `limit` parameter in the `AudioDataset` class.

## 5. Debugging
You can set breakpoints in any `.py` file (like `main.py` or `src/train.py`) by clicking to the left of the line numbers. When you run the project using the **Run and Debug** sidebar, VS Code will pause execution at these points, allowing you to inspect variables and state.
