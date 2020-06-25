import requests
import json
import csv

item_list = list(range(554, 567, 1))

csv_name = "rune_data"

full_dict = {}
labels = ['timestamp']

for itemID in item_list:
    r = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/graph/{}.json'.format(str(itemID)))
    json_data = json.loads(r.text)

    current_daily_dict = json_data['daily']

    for daily_timestamp in current_daily_dict:
        if( daily_timestamp in full_dict ):
            full_dict[daily_timestamp].append(current_daily_dict[daily_timestamp])
        else:
            full_dict[daily_timestamp] = [current_daily_dict[daily_timestamp]]

    r2 = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/catalogue/detail.json?item=' + str(itemID))
    labels.append(json.loads(r2.text)['item']['name'].replace(" ", "_"))


# write to csv

with open('data/{}.csv'.format(csv_name), mode='w', newline='') as GE_data:
    GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    GE_writer.writerow(labels)

    for daily_timestamp in full_dict:
        new_array = [daily_timestamp]
        new_array.extend(full_dict[daily_timestamp])
        GE_writer.writerow(new_array)