# youtube-history-app
A small app that allow show you some stats based on the youtube history provided to it

## Getting Started

You can get your Youtube watch history [HERE](https://takeout.google.com/settings/takeout). Make sure to download it in the .JSON format.

Before you install the requirements, make sure to create a new virtual environment, like explained [here](https://realpython.com/python-virtual-environments-a-primer/), activate it and then run 'pip install -r requirements.txt'

After having the dependencies install, you can start the app with the command 'streamlit run app.py'.

## Known Issues

The app is using the pytube package and as such the retrival of the extra data needed from Youtube is very slow.
