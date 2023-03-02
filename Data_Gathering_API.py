import requests
import pandas as pd

page_size = 10
headers = {
    'X-Api-Key': 'XXXXXXX',
}

params = (
    ('page_size', str(page_size)),
    ('limit_orders', '99999'),
)
next_url = 'https://us.market-api.kaiko.io/v1/data/order_book_snapshots.latest/exchanges/krkn/spot/btc-usd/snapshots/full'
collected_data = pd.DataFrame()

for thing in range(0, 2):
    response = requests.get(next_url, headers=headers, params=params)
    rjson = response.json()
    next_url = rjson['next_url']

    for item in range(0, page_size):
        date = response.json()['data'][item]['poll_timestamp']
        asks = pd.DataFrame.from_dict(response.json()['data'][item]['asks'], orient='columns')
        asks['type'] = 'a'
        asks['date'] = date
        asks = asks.set_index('date', drop=True)

        bids = pd.DataFrame.from_dict(response.json()['data'][item]['bids'], orient='columns')
        bids['type'] = 'b'
        bids['date'] = date
        bids = bids.set_index('date', drop=True)
        collected_data = pd.concat([collected_data, bids, asks])

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 4));
plt.title('Distribution of time difference between consecutive datapoints');
plt.xlabel('Seconds')
plt.ylabel('Number of occurence')
plt.hist(
    pd.Series(list(collected_data.index.unique())).sort_values(ascending=True).map(lambda x: round(x / 1000)).diff(),
    300)
