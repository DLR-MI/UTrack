
def reassign_ucmc(args, video_name):
    
    # from https://github.com/corfyi/UCMCTrack/blob/e2ad836216c8edba5f4166c80e5cff9e042380d5/run_mot17_test.bat
    if 'MOT17-01' in video_name:
        args.yolo.conf = 0.5
        args.tracker.ucmc.wx = 0.20
        args.tracker.ucmc.wy = 0.40
        args.tracker.ucmc.vmax = 0.5
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
        args.tracker.camera_motion.method = None
    elif 'MOT17-02' in video_name:
        args.tracker.ucmc.wx = 0.10
        args.tracker.ucmc.wy = 0.20
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-03' in video_name:
        args.yolo.conf = 0.4
        args.tracker.ucmc.wx = 0.45
        args.tracker.ucmc.wy = 0.55
        args.tracker.ucmc.vmax = 0.5
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 12 
        args.tracker.thresholds.matching.distance[1] = 12 
        args.tracker.camera_motion.method = None
    elif 'MOT17-04' in video_name:
        args.tracker.ucmc.wx = 0.5
        args.tracker.ucmc.wy = 0.5
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-05' in video_name:
        args.tracker.ucmc.wx = 0.10
        args.tracker.ucmc.wy = 0.20
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-06' in video_name:
        args.yolo.conf = 0.4
        args.tracker.ucmc.wx = 0.10
        args.tracker.ucmc.wy = 5.00
        args.tracker.ucmc.vmax = 1.0
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 19 
        args.tracker.thresholds.matching.distance[1] = 19 
    elif 'MOT17-07' in video_name:
        args.yolo.conf = 0.5
        args.tracker.ucmc.wx = 2.00
        args.tracker.ucmc.wy = 2.00
        args.tracker.ucmc.vmax = 0.5
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-08' in video_name:
        args.yolo.conf = 0.25
        args.tracker.ucmc.wx = 0.70
        args.tracker.ucmc.wy = 1.50
        args.tracker.ucmc.vmax = 3.0
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 9 
        args.tracker.thresholds.matching.distance[1] = 9 
    elif 'MOT17-09' in video_name:
        args.tracker.ucmc.wx = 0.5
        args.tracker.ucmc.wy = 1.0
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-10' in video_name:
        args.tracker.ucmc.wx = 5.0
        args.tracker.ucmc.wy = 5.0
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 60 
        args.tracker.thresholds.matching.distance[1] = 60 
    elif 'MOT17-11' in video_name:
        args.tracker.ucmc.wx = 5.0
        args.tracker.ucmc.wy = 5.0
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 40 
        args.tracker.thresholds.matching.distance[1] = 40 
    elif 'MOT17-12' in video_name:
        args.yolo.conf = 0.5
        args.tracker.ucmc.wx = 0.50
        args.tracker.ucmc.wy = 0.50
        args.tracker.ucmc.vmax = 0.5
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    elif 'MOT17-13' in video_name:
        args.tracker.ucmc.wx = 5.0
        args.tracker.ucmc.wy = 5.0
        args.tracker.track_buffer = 10
        args.tracker.thresholds.matching.distance[0] = 40 
        args.tracker.thresholds.matching.distance[1] = 40 
    elif 'MOT17-14' in video_name:
        args.yolo.conf = 0.4
        args.tracker.ucmc.wx = 1.00
        args.tracker.ucmc.wy = 1.00
        args.tracker.ucmc.vmax = 1.0
        args.tracker.track_buffer = 21
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
    # from https://github.com/corfyi/UCMCTrack/blob/e2ad836216c8edba5f4166c80e5cff9e042380d5/run_mot20_test.bat
    elif 'MOT20-04' in video_name:
        args.yolo.conf = 0.43
        args.tracker.ucmc.wx = 0.5
        args.tracker.ucmc.wy = 0.4
        args.tracker.ucmc.vmax = 0.6
        args.tracker.frame_rate = 25
        args.tracker.track_buffer = 35
        args.tracker.thresholds.matching.distance[0] = 10 
        args.tracker.thresholds.matching.distance[1] = 10 
        args.tracker.camera_motion.method = None
    elif 'MOT20-06' in video_name:
        args.yolo.conf = 0.30
        args.tracker.ucmc.wx = 0.2
        args.tracker.ucmc.wy = 0.4
        args.tracker.ucmc.vmax = 0.5
        args.tracker.frame_rate = 25
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 9 
        args.tracker.thresholds.matching.distance[1] = 9 
        args.tracker.camera_motion.method = None
    elif 'MOT20-07' in video_name:
        args.yolo.conf = 0.43
        args.tracker.ucmc.wx = 0.4
        args.tracker.ucmc.wy = 0.3
        args.tracker.ucmc.vmax = 0.6
        args.tracker.frame_rate = 25
        args.tracker.track_buffer = 35
        args.tracker.thresholds.matching.distance[0] = 12 
        args.tracker.thresholds.matching.distance[1] = 12
        args.tracker.camera_motion.method = None
    elif 'MOT20-08' in video_name:
        args.yolo.conf = 0.33
        args.tracker.ucmc.wx = 0.2
        args.tracker.ucmc.wy = 0.4
        args.tracker.ucmc.vmax = 0.5
        args.tracker.frame_rate = 25
        args.tracker.track_buffer = 30
        args.tracker.thresholds.matching.distance[0] = 9 
        args.tracker.thresholds.matching.distance[1] = 9
        args.tracker.camera_motion.method = None