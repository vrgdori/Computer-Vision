import asyncio
import cv2
import json
import time
from camera_stream import AsyncVideoStream
from detector import ObstacleDetector
import config

async def main():
    vs = AsyncVideoStream(src=0).start()
    detector = ObstacleDetector()
    
    print(json.dumps({"status": "SYSTEM_READY", "config": {"frame_skip": config.FRAME_SKIP_N}}))
    
    frame_count = 0
    
    try:
        while True:
            frame = vs.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            
            if frame_count % config.FRAME_SKIP_N == 0:
                resized_frame = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
                
                start_time = time.time()
                obstacles = detector.process_frame(resized_frame)
                process_time_ms = round((time.time() - start_time) * 1000, 2)

                output_payload = {
                    "timestamp": time.time(),
                    "frame_id": frame_count,
                    "processing_time_ms": process_time_ms,
                    "obstacle_count": len(obstacles),
                    "obstacles": obstacles
                }

                print(json.dumps(output_payload, indent=2))

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print(json.dumps({"status": "SYSTEM_STOPPING"}))
    finally:
        vs.stop()

if __name__ == "__main__":
    asyncio.run(main())