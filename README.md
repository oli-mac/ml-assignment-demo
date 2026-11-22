# Machine Learning Assignment 1

Run both trained models in one Streamlit app. Use the example connections on the classification tab. On the regression tab, change a sensor reading and select **Reset to example** to restore the held-out reading.

## Local preview

From this folder, using Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Public instructor demo

1. Use this folder as the root of a dedicated GitHub repository. Include streamlit_app.py, features.py, requirements.txt, and every file in models/.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**, choose the repository and branch, and enter **streamlit_app.py** as the entry point. Under **Advanced settings**, select Python 3.12 and deploy.
3. Enable public viewing. Open the assigned app URL in a signed-out browser. Try the example connections, change a turbine sensor, and reset the reading. Check the controls on a phone.
4. Once verified, hyperlink **Open interactive demo** on slide 20 to that URL. Include the same link with both notebooks and the PowerPoint in Classroom when ready.

The app uses the saved pipelines in models/. When retraining, copy each new model together with its metadata and examples. Match the deployed scikit-learn version to the version used for training.
