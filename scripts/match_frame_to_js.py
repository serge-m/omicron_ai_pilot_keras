import pandas
import csv
import os

js4_data = pandas.read_csv("./data/data_js.csv")
image_data = pandas.read_csv("./data/data_image.csv")

js4_data.sort_values("timestamp")
image_data.sort_values("timestamp")

with open(os.path.join("./data", 'data.csv'), 'w') as writeFile:
	writer = csv.writer(writeFile)
	iter = js4_data.iterrows()
	data = None

	search = True
	for _, image_row in image_data.iterrows():
		while(search):
			# in case if we have more images than signals at the end
			try:
				data = next(iter)[1]
			except:
				search = False

			if image_row["timestamp"] < data["timestamp"]:
				break

		writer.writerow([image_row["image"],data["msg.drive.steering_angle"],data["msg.drive.speed"]])
