import asyncio
import cv2
import json
import time
from camera_stream import AsyncVideoStream
from detector import ObstacleDetector
import config
import csv
import os
import statistics


async def main():
    vs = AsyncVideoStream().start()
    detector = ObstacleDetector()

    os.makedirs("./results", exist_ok=True)

    csv_path = "./results/performance_log.csv"
    summary_path = "./results/performance_summary.json"

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")

    fieldnames = [
        "timestamp",
        "frame_id",
        "video_timestamp_ms",
        "resize_time_ms",
        "processing_time_ms",
        "total_pipeline_time_ms",
        "processing_fps",
        "processing_interval_ms",
        "obstacle_count"
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    print(json.dumps({
        "status": "SYSTEM_READY",
        "source_type": config.INPUT_TYPE,
        "source": str(config.INPUT_SOURCE),
        "config": {"frame_skip": config.FRAME_SKIP_N}
    }))

    processing_times = []
    last_processed_frame_id = 0
    last_processing_start = None

    try:
        while not vs.is_finished:

            frame, frame_id, video_timestamp_ms = vs.read_with_metadata()

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Ugyanazt a képkockát ne dolgozzuk fel többször
            if frame_id == last_processed_frame_id:
                await asyncio.sleep(0.001)
                continue

            last_processed_frame_id = frame_id

            # Frame skipping az eredeti képkockaszám alapján
            if frame_id % config.FRAME_SKIP_N != 0:
                await asyncio.sleep(0.001)
                continue

            # Teljes pipeline kezdete
            pipeline_start = time.perf_counter()

            # Átméretezés mérése
            resize_start = time.perf_counter()

            resized_frame = cv2.resize(
                frame,
                (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
            )

            resize_time_ms = (
                time.perf_counter() - resize_start
            ) * 1000

            # Akadályfelismerés mérése
            processing_start = time.perf_counter()

            if last_processing_start is None:
                processing_interval_ms = None
            else:
                processing_interval_ms = (
                    processing_start - last_processing_start
                ) * 1000

            last_processing_start = processing_start

            obstacles = detector.process_frame(resized_frame)

            processing_end = time.perf_counter()

            processing_time_ms = (
                processing_end - processing_start
            ) * 1000

            total_pipeline_time_ms = (
                processing_end - pipeline_start
            ) * 1000

            processing_fps = (
                1000 / processing_time_ms
                if processing_time_ms > 0 else 0
            )

            processing_times.append(processing_time_ms)

            # Mérési adatok
            output_payload = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "video_timestamp_ms": video_timestamp_ms,
                "resize_time_ms": round(resize_time_ms, 3),
                "processing_time_ms": round(processing_time_ms, 3),
                "total_pipeline_time_ms": round(total_pipeline_time_ms, 3),
                "processing_fps": round(processing_fps, 2),
                "processing_interval_ms": (
                    round(processing_interval_ms, 3)
                    if processing_interval_ms is not None else None
                ),
                "obstacle_count": len(obstacles),
                "obstacles": obstacles
            }

            print(json.dumps(output_payload, indent=2))

            # CSV mentés
            writer.writerow({
                "timestamp": output_payload["timestamp"],
                "frame_id": frame_id,
                "video_timestamp_ms": video_timestamp_ms,
                "resize_time_ms": round(resize_time_ms, 3),
                "processing_time_ms": round(processing_time_ms, 3),
                "total_pipeline_time_ms": round(total_pipeline_time_ms, 3),
                "processing_fps": round(processing_fps, 2),
                "processing_interval_ms": (
                    round(processing_interval_ms, 3)
                    if processing_interval_ms is not None else ""
                ),
                "obstacle_count": len(obstacles)
            })

            csv_file.flush()

            # Előnézet
            if config.SHOW_PREVIEW:
                vis_frame = detector.draw_detections(
                    resized_frame, obstacles
                )

                cv2.imshow("Drone Obstacle Detection", vis_frame)

                key = cv2.waitKey(1) & 0xFF

                if key != 255 and key != 0:
                    print(json.dumps({
                        "status": "USER_INTERRUPT",
                        "key_pressed": key
                    }))
                    break

            await asyncio.sleep(0.001)

        print(json.dumps({"status": "PROCESSING_COMPLETE"}))

    except KeyboardInterrupt:
        print(json.dumps({"status": "SYSTEM_STOPPING"}))

    finally:
        vs.stop()
        csv_file.close()
        cv2.destroyAllWindows()

        # Statisztikai összesítés
        if processing_times:
            sorted_times = sorted(processing_times)

            p95_index = int(0.95 * (len(sorted_times) - 1))

            summary = {
                "processed_frames": len(processing_times),
                "average_processing_time_ms": round(
                    statistics.mean(processing_times), 3
                ),
                "median_processing_time_ms": round(
                    statistics.median(processing_times), 3
                ),
                "min_processing_time_ms": round(
                    min(processing_times), 3
                ),
                "max_processing_time_ms": round(
                    max(processing_times), 3
                ),
                "p95_processing_time_ms": round(
                    sorted_times[p95_index], 3
                ),
                "average_processing_fps": round(
                    1000 / statistics.mean(processing_times), 2
                )
            }

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)

            print(json.dumps({
                "status": "PERFORMANCE_SUMMARY",
                **summary
            }, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
