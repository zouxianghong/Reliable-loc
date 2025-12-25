#!/bin/bash

###############SGV Scoring analysis###############
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 500 --end_idx 600 --init_type gt_xy --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 500 --end_idx 600 --init_type gt_xy --use_sgv --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 500 --end_idx 600 --init_type gt_xy --use_sgv --use_yaw --loc_mode 0 --test_name scoring && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1000 --end_idx 1100 --init_type gt_xy --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1000 --end_idx 1100 --init_type gt_xy --use_sgv --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1000 --end_idx 1100 --init_type gt_xy --use_sgv --use_yaw --loc_mode 0 --test_name scoring && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1500 --end_idx 1600 --init_type gt_xy --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1500 --end_idx 1600 --init_type gt_xy --use_sgv --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 1500 --end_idx 1600 --init_type gt_xy --use_sgv --use_yaw --loc_mode 0 --test_name scoring && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 2000 --end_idx 2100 --init_type gt_xy --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 2000 --end_idx 2100 --init_type gt_xy --use_sgv --loc_mode 0 --test_name scoring && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 2000 --end_idx 2100 --init_type gt_xy --use_sgv --use_yaw --loc_mode 0 --test_name scoring && \

###############Param Analysis###############
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.001 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.002 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.004 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.008 --loc_mode 1 --test_name param && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.016 --loc_mode 1 --test_name param && \

###############Necessity of switching the loc mode###############
python monte_carlo_loc/main.py --dataset info_campus --start_idx 4250 --end_idx 5000 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 --test_name necessity && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 4250 --end_idx 5000 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name necessity && \

python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 6500 --end_idx 7000 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 --test_name necessity && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 6500 --end_idx 7000 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name necessity && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 3000 --end_idx 3500 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 --test_name necessity && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 3000 --end_idx 3500 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode 1 --test_name necessity && \

python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 1750 --end_idx 2250 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode -1 --test_name necessity && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 1750 --end_idx 2250 --init_type gt_xy_yaw --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode 1 --test_name necessity && \

###############Quantitative Evaluation###############
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset cs_college --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 && \

python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset info_campus --start_idx 300 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 && \

python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset zhongshan_park --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 && \

python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset jiefang_road --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1 && \

python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road1 --start_idx 400 --use_sgv --use_yaw --min_reliable_value 0.0001 --loc_mode -1 && \

python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --loc_mode 0 && \
python monte_carlo_loc/main.py --dataset yanjiang_road2 --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1