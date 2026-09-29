import asyncio
import cv2
import json
import time

from camera_stream import AsyncVideoStream
from detector import ObstacleDetector
import config

async def main():
    vs = AsyncVideoStream().start()
    detector = ObstacleDetector()
    
    print(json.dumps({
            "status": "SYSTEM_READY", 
            "source_type": config.INPUT_TYPE,
            "source": str(config.INPUT_SOURCE),
            "config": {"frame_skip": config.FRAME_SKIP_N}
        }))   
     
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


                if config.SHOW_PREVIEW:
                    vis_frame = detector.draw_detections(resized_frame, obstacles)
                    cv2.imshow("Drone Obstacle Detection", vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(json.dumps({"status": "USER_INTERRUPT"}))
                        break

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print(json.dumps({"status": "SYSTEM_STOPPING"}))
    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())