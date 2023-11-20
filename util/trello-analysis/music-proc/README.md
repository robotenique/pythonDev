# Trello-analysis/music-proc

This folder has a quick script and corresponding notebook that will perform the following main tasks:
- Extract the 'cards' and 'list' of a Trello exported data
- Since the data itself is incomplete, we manually query the Trello API to get the correspoding "created_at" for all cards
- In the notebook, using the processed objects to perform various queries that summarize the data (Top cards and worst cards by different segmentations)



**Requirements**:

- Install the requirements with `pip install -r requirements.txt`
- Download the Trello data from the board through: Menu -> Print, export, and share -> Export as JSON. This should be saved on the same folder;
- Create a yaml file `trello_config.yaml` with the following structure:
```yaml
api_key: <your_api_key>
api_secret: <your_api_secret>
```
- On the script `trello_music_board_processor.py`, change the variable `json_file_path` to the path of the downloaded json file;
- Run the script `trello_music_board_processor.py` to test if everything is working;
- If everything is working, you can go to TrelloMusicVisualizer.ipynb and run the notebook to get the results.

**Notes**:


You can get a Trello API key and token through the Power-ups and Integrations of Trello [here](https://trello.com/power-ups/admin).