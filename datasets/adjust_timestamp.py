# for self-collected data
import os
import shutil
import pandas as pd

from utils.util import check_makedirs

def timestamp2str(timestamp):
    if not isinstance(timestamp, float):
        return str(timestamp)
    stamp_int = int(timestamp)
    stamp_float = int((timestamp - stamp_int + 5.e-7) * 1000000)
    return str(stamp_int) + "." + str(stamp_float).zfill(6)

# data dir
data_dir = '/home/ericxhzou/Data/benchmark_datasets/Self_Collected/For_Training'
trip_dir = 'helmet_submap3'
submap_type = 'pointcloud_30m_0.5m'
old_cloud_dir = os.path.join(data_dir, trip_dir, submap_type)
new_cloud_dir = os.path.join(data_dir, trip_dir, f'{submap_type}_new')
check_makedirs(new_cloud_dir)

# csv
centroid_csv = os.path.join(data_dir, trip_dir, f'{submap_type}.csv')
trip_records = pd.read_csv(centroid_csv, sep=',')

pose_csv = os.path.join(data_dir, trip_dir, f'pose.csv')
pose_df = pd.read_csv(pose_csv, sep=',')

for index, row in trip_records.iterrows():
    old_stamp = timestamp2str(int(row['timestamp']))
    new_stamp = '3' + old_stamp[1:]
    trip_records.loc[index, 'timestamp'] = new_stamp
    pose_df.loc[index, 'timestamp'] = new_stamp
    # copy bin file
    old_bin = os.path.join(old_cloud_dir, old_stamp + '.bin')
    new_bin = os.path.join(new_cloud_dir, new_stamp + '.bin')
    shutil.copyfile(old_bin, new_bin)

# save new centroid csv
new_centorid_df = pd.DataFrame(columns=['timestamp', 'northing', 'easting'])
new_centorid_df = trip_records.sort_values('timestamp')  # sort by time stamp
new_centorid_df.reset_index(drop=True)
new_csv = os.path.join(data_dir, trip_dir, f'{submap_type}_new.csv')
new_centorid_df.to_csv(new_csv, index=False)

# save new pose csv
new_pose_df = pd.DataFrame(columns=['timestamp','T00', 'T01', 'T02', 'T03', 'T10', 'T11', 'T12', 'T13', 'T20', 'T21', 'T22', 'T23'])
new_pose_df = pose_df.sort_values('timestamp')  # sort by time stamp
new_pose_df.reset_index(drop=True)
new_csv = os.path.join(data_dir, trip_dir, f'pose_new.csv')
new_pose_df.to_csv(new_csv, index=False)