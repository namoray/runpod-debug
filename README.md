# runpod-debug


Try to run the load_model.py script. After the initial download of stuff (which is fine), take note of the loading time of the model.

On good hardware, this takes ~5s to load the whole model. On bad hardware (is my theory), it takes 60 seconds. Why?

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python load_model.py
```