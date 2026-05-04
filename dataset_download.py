from cryoet_data_portal import Client, Dataset

client = Client()

dataset = Dataset.get_by_id(client, 10440)
dataset.download_everything()