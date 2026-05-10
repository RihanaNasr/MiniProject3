# Mini Project 3 – Real-Time Recommendation System

**Domain:** Movies  
**Dataset:** MovieLens 1M (1,000,209 ratings, 6,040 users, ~3,900 movies)  
**Focus:** Real-Time Intelligence (trend detection, alerts, dynamic recommendations)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset Selection & Justification](#3-dataset-selection--justification)
4. [Machine Learning Component (Batch Processing)](#4-machine-learning-component-batch-processing)
5. [Streaming Component (Kafka + Spark)](#5-streaming-component-kafka--spark)
6. [Real-Time Analytics & Custom Metric](#6-real-time-analytics--custom-metric)
7. [Alert System & Late Data Handling](#7-alert-system--late-data-handling)
8. [ML + Streaming Integration (Dynamic Recommendations)](#8-ml--streaming-integration-dynamic-recommendations)
9. [Results & Sample Outputs](#9-results--sample-outputs)
10. [Performance Evaluation](#10-performance-evaluation)
11. [Challenges & Lessons Learned](#11-challenges--lessons-learned)
12. [Run Instructions](#12-run-instructions)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Project Overview

This project implements an **end-to-end real-time recommendation system** using Apache Spark and Kafka. The system:

- Learns user preferences from historical MovieLens 1M ratings using ALS collaborative filtering.
- Processes live user interactions via a Kafka streaming pipeline.
- Generates dynamic top-5 recommendations per user in real time.
- Detects trending items and triggers alerts based on rating spikes.

### Selected Focus: **Real-Time Intelligence**
- Trending score detection (`interactions × avg_rating`)
- Real-time alerts for trending items and activity spikes
- Sub‑5‑second recommendation latency

---

## 2. System Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6a19c201-9f8e-4e8b-98b3-4f943acf2d12" />


### Component Roles

| Component | Technology | Purpose |
|-----------|------------|---------|
| Batch ML | Spark ALS | Train recommendation model on historical data |
| Message Queue | Kafka (2 partitions) | Stream real-time user ratings |
| Stream Processing | Spark Structured Streaming | Windowed analytics, alerts, recommendations |
| Storage | JSON files | Save results for dashboard & report |
| Producer | Python (kafka-python) | Simulate real-time data stream |

---

## 3. Dataset Selection & Justification

### Dataset: MovieLens 1M

| Property | Value |
|----------|-------|
| Source | GroupLens (https://grouplens.org/datasets/movielens/1m/) |
| Size | 1,000,209 ratings |
| Users | 6,040 |
| Movies | ~3,900 |
| Format | `user_id::movie_id::rating::timestamp` |
| Rating scale | 1–5 (whole numbers only) |
| Timespan | 2000–2003 (simulated timestamps) |

### Why This Dataset?

1. **Size & Density:** With 1M ratings, the user‑item matrix is sufficiently dense for collaborative filtering. ALS can learn meaningful latent factors (rank=10 works well).

2. **Distributed Processing Justification:**  
   - The full 6,040 × 3,900 matrix would be ~23 million cells – storing this in memory on a single machine is inefficient.  
   - Spark distributes the ALS computation across cores, enabling fast training (<1 minute on 4GB VM).

3. **Challenges:**  
   - Timestamps are in seconds (needs conversion for watermarking).  
   - Ratings are discrete 1–5; streaming simulation uses current epoch time.

### Data Sample
```
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
2::1193::5::978300760
```

---

## 4. Machine Learning Component (Batch Processing)

### 4.1 Data Preprocessing

```python
df = spark.read.option("delimiter", "::").csv(data_path, schema=schema)
df = df.dropna(subset=["user_id", "item_id", "rating"])
df = df.filter((df.rating >= 0.5) & (df.rating <= 5.0))
```

- Removed 0 invalid records (all ratings are valid).
- Final clean records: **1,000,209**

### 4.2 ALS Model Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `rank` | 10 | Latent factors – balances accuracy & speed |
| `maxIter` | 10 | Converges within 10 iterations for this dataset |
| `regParam` | 0.1 | Standard regularization to prevent overfitting |
| `coldStartStrategy` | "drop" | Avoids evaluator errors on unseen users/items |
| `nonnegative` | True | Prevents negative ratings in predictions |

### 4.3 Train/Test Split

- **Training:** 80% (800,167 records)
- **Test:** 20% (200,042 records)

### 4.4 Evaluation & Tuning

**Baseline RMSE:** `0.8765`

Since 0.8765 < 1.5, no hyperparameter tuning was applied.  
The code includes an automatic tuning phase that would trigger if RMSE > 1.5, adjusting `rank`, `maxIter`, and `regParam`.

### 4.5 Model Persistence

```python
model.write().overwrite().save(model_path)
```

Model saved to `models/als_model` – loaded later by streaming application.

---

## 5. Streaming Component (Kafka + Spark)

### 5.1 Kafka Setup

**Version:** Kafka 3.4.0 (Scala 2.12) – standalone installation

| Component | Port | Purpose |
|-----------|------|---------|
| Zookeeper | 2182 | Cluster coordination |
| Kafka Broker | 9093 | Message broker (non‑default port to avoid Hive conflict) |

**Topic Configuration:**
```bash
kafka-topics.sh --create --topic movie-ratings \
  --bootstrap-server localhost:9093 \
  --partitions 2 \
  --replication-factor 1
```

### 5.2 Partitioning Strategy

**Why partition by `item_id`?**  
- All ratings for the same movie go to the same partition.  
- Spark streaming can then aggregate per‑item metrics **without shuffling data** across partitions.  
- This significantly reduces network overhead and latency.

**Producer code:**
```python
partition_key = str(record["item_id"])
producer.send(TOPIC, key=partition_key, value=record)
```

### 5.3 Kafka Producer (Simulated Real‑Time)

```python
def simulate_stream(dataset_path):
    for idx, row in df.iterrows():
        record = {
            "user_id": int(row['user_id']),
            "item_id": int(row['item_id']),
            "rating": float(row['rating']),
            "timestamp": int(time.time())  # Use current time
        }
        producer.send(TOPIC, key=str(record["item_id"]), value=record)
        time.sleep(0.01)  # 100 events/sec
```

**Output sample:**
```
[100] Sent: User 2 -> Item 1690 (Rating: 3.0) | Partition: 1
[200] Sent: User 3 -> Item 1580 (Rating: 3.0) | Partition: 1
[300] Sent: User 5 -> Item 2858 (Rating: 4.0) | Partition: 0
```

### 5.4 Spark Structured Streaming Configuration

```python
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9093") \
    .option("subscribe", "movie-ratings") \
    .option("startingOffsets", "latest") \
    .load()

parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*") \
    .filter(col("user_id").isNotNull() & col("item_id").isNotNull()) \
    .withColumn("event_time", current_timestamp()) \
    .withWatermark("event_time", "2 minutes")
```

---

## 6. Real-Time Analytics & Custom Metric

### 6.1 Windowed Aggregations

**Window:** 30 seconds, slide: 10 seconds

```python
windowed_analytics = parsed_df.groupBy(
    window(col("event_time"), "30 seconds", "10 seconds"),
    col("item_id")
).agg(
    avg("rating").alias("avg_rating"),
    count("*").alias("interactions_count"),
    variance("rating").alias("rating_variance"),
    (count("*") * avg("rating")).alias("trending_score")
)
```

### 6.2 Custom Metric: Trending Score

**Formula:** `trending_score = interactions × avg_rating`

| Item | Example | Score |
|------|---------|-------|
| High rating, low interactions | Rating 5.0, 1 interaction | 5.0 |
| Medium rating, high interactions | Rating 3.5, 10 interactions | 35.0 |
| Both high | Rating 4.6, 7 interactions | **32.0** (trending) |

**Why this metric?**  
- Traditional "average rating" can be skewed by a single 5-star vote.  
- Trending score ensures that items gaining **momentum** (many interactions) are recognised even if average rating is slightly lower.

**Real example from output:**
```
Item 608: Score=32.0, Rating=4.57, Interactions=7
Item 1136: Score=28.0, Rating=4.67, Interactions=6
```

---

## 7. Alert System & Late Data Handling

### 7.1 Alert Conditions

| Type | Condition | Example |
|------|-----------|---------|
| **Trending Alert** | `avg_rating > 4.5 AND interactions ≥ 3` | `🔥 ALERT: Item 1136 is TRENDING! Rating: 4.67, Interactions: 6` |
| **Spike Alert** | `interactions ≥ 10` | `⚡ ALERT: Activity SPIKE on Item 2858! 17 interactions` |

### 7.2 Alert Implementation

```python
if avg_rating > 4.5 and interactions >= 3:
    alerts.append(f"🔥 ALERT: Item {item_id} is TRENDING! Rating: {avg_rating:.2f}, Interactions: {interactions}")
elif interactions >= 10:
    alerts.append(f"⚡ ALERT: Activity SPIKE on Item {item_id}! {interactions} interactions")
```

### 7.3 Late Data Handling (Watermark)

```python
.withWatermark("event_time", "2 minutes")
```

**How it works:**  
- The watermark tracks the maximum observed event time minus 2 minutes.  
- Any event arriving after this threshold is **dropped**.  
- This prevents state explosion and ensures window results finalise.

**Justification:** 2 minutes is reasonable for a simulated stream with minor network delays. In production, this would be tuned based on expected maximum lateness.

---

## 8. ML + Streaming Integration (Dynamic Recommendations)

### 8.1 Approach

Instead of joining streaming data with the entire model (which is expensive), we:

1. Extract distinct `user_id`s from the current micro‑batch.
2. Call `model.recommendForUserSubset(users_df, 5)`.
3. Return top‑5 items for those users.

```python
def process_recommendation_batch(df, epoch_id):
    if df.count() > 0:
        users_df = df.select("user_id").distinct()
        if users_df.count() > 0:
            recommendations = model.recommendForUserSubset(users_df, 5)
            for row in recommendations.collect():
                user_id = row['user_id']
                rec_items = [r['item_id'] for r in row['recommendations']]
                print(f"🎯 RECOMMENDATION for User {user_id}: Top 5 Items: {rec_items}")
```

### 8.2 Example Recommendations

```
🎯 RECOMMENDATION for User 48: Top 5 Items: [572, 1851, 318, 260, 1198]
🎯 RECOMMENDATION for User 10: Top 5 Items: [572, 2197, 1851, 2342, 406]
🎯 RECOMMENDATION for User 1: Top 5 Items: [572, 1851, 527, 318, 1207]
```

### 8.3 Latency (Bonus Requirement)

**Measured latency: < 5 seconds**

- Micro‑batch trigger interval is 5 seconds.
- `recommendForUserSubset` executes only on the subset of users in the batch.
- For typical batch sizes (10–100 users), computation takes <100ms.
- End-to-end latency from Kafka ingestion to recommendation output: **2–4 seconds**.

---

## 9. Results & Sample Outputs

### 9.1 Model RMSE

```
============================================================
Loading batch data for ML training...
============================================================
✓ Loaded 1,000,209 records from dataset

--- Data Preprocessing ---
✓ Removed 0 invalid records
✓ Final clean records: 1,000,209

--- Splitting Dataset ---
✓ Training set: 800,167 records
✓ Test set: 200,042 records

--- Training Baseline ALS Model ---
✓ Baseline model training complete

--- Evaluating Baseline Model ---
✓ Baseline RMSE = 0.8765

✓ Baseline RMSE is good (< 1.5), no tuning needed

--- Saving Model ---
✓ Model saved to /home/hduser/MiniProject3/models/als_model

============================================================
Batch ML Pipeline Completed Successfully!
============================================================
```

### 9.2 Trending Items (Console Output)

```
📊 Top Trending Items:
  Item 608: Score=32.0, Rating=4.57, Interactions=7
  Item 1136: Score=28.0, Rating=4.67, Interactions=6
  Item 457: Score=22.0, Rating=4.40, Interactions=5
```

### 9.3 Alerts

```
🔥 ALERT: Item 1136 is TRENDING! Rating: 4.67, Interactions: 6
🔥 ALERT: Item 608 is TRENDING! Rating: 4.57, Interactions: 7
⚡ ALERT: Activity SPIKE on Item 2858! 17 interactions
⚡ ALERT: Activity SPIKE on Item 1196! 12 interactions
```

### 9.4 Saved JSON Files

**`dashboard_analytics.json`** – Example entry:
```json
{
  "metrics": [
    {
      "item_id": 608,
      "avg_rating": 4.57,
      "interactions": 7,
      "trending_score": 32.0,
      "timestamp": 1778435298.3319652
    }
  ],
  "alerts": [
    "🔥 ALERT: Item 1136 is TRENDING! Rating: 4.67, Interactions: 6",
    "🔥 ALERT: Item 608 is TRENDING! Rating: 4.57, Interactions: 7"
  ],
  "last_update": "2026-05-10T20:48:18.332109"
}
```

**`dashboard_recs.json`** – Example:
```json
{
  "recent_recommendations": [
    {
      "user_id": 48,
      "recommendations": [572, 1851, 318, 260, 1198]
    },
    {
      "user_id": 10,
      "recommendations": [572, 2197, 1851, 2342, 406]
    }
  ]
}
```

### 9.5 Trending Items CSV

| item_id | avg_rating | interactions | trending_score | timestamp |
|---------|------------|--------------|----------------|-----------|
| 608 | 4.57 | 7 | 32.0 | 1778435298.331965 |
| 1136 | 4.67 | 6 | 28.0 | 1778435298.331961 |
| 608 | 4.60 | 5 | 23.0 | 1778435298.332041 |
| 457 | 4.40 | 5 | 22.0 | 1778435298.331979 |
| 3751 | 4.40 | 5 | 22.0 | 1778435298.332050 |

---

## 10. Performance Evaluation

### 10.1 Latency Measurement

| Component | Time |
|-----------|------|
| Kafka message ingest | <10 ms |
| Spark window trigger | 5 seconds (configurable) |
| `recommendForUserSubset` | <100 ms (for 10–100 users) |
| **Total end‑to‑end latency** | **2–4 seconds** |

**Bonus achieved:** Recommendations generated with <5 seconds latency.

### 10.2 Throughput

- **Producer:** 100 events/second (can be increased by reducing `time.sleep()`)
- **Spark streaming:** Processes micro‑batches every 5 seconds; each batch handles up to 500 events comfortably on 4GB VM.
- **Kafka topic:** 2 partitions, balanced distribution by `item_id`.

### 10.3 Resource Usage

| Resource | Used |
|----------|------|
| CPU | 2–3 cores (Spark parallelisation) |
| Memory | ~3 GB (Spark driver + executor) |
| Disk | 200 MB (logs, checkpoints, model) |

---

## 11. Challenges & Lessons Learned

### 11.1 Technical Challenges & Solutions

| Challenge | Root Cause | Solution |
|-----------|------------|----------|
| Zookeeper refused to start | Log4J conflict with Hive libraries | Unset `CLASSPATH` and related env vars; used custom ports (2182, 9093) |
| Kafka `NoSuchMethodError` | Scala version mismatch (Kafka 3.6.0 needed Scala 2.13, system had 2.12) | Downgraded to Kafka 3.4.0 with Scala 2.12 |
| `Failed to find data source: kafka` | Missing Spark SQL Kafka package | Used `spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0` |
| Streaming app showed no output | `startingOffsets="latest"` with producer finished earlier | Changed to `"earliest"` or restarted producer after streaming app |
| IndentationError in `streaming_app.py` | Manual copy‑paste errors | Rewrote file with proper indentation |
| Port conflicts (2181, 9092) | Hive/Zookeeper using default ports | Used ports 2182 (Zookeeper) and 9093 (Kafka) |

### 11.2 Key Lessons Learned

1. **Classpath isolation is critical** when multiple big data components coexist. Creating a standalone Kafka installation with clean environment variables resolves many conflicts.

2. **Spark streaming watermarking** prevents state explosion and handles late data – essential for production systems.

3. **Partitioning by key (`item_id`)** eliminates shuffles for per‑item aggregations, drastically improving performance.

4. **`recommendForUserSubset`** is highly efficient – real‑time recommendations for active users only, not the entire user base.

5. **Simulating real‑time data** with `time.sleep()` works well for demos, but production would use a true event source (e.g., web logs).

---

## 12. Run Instructions

### 12.1 Prerequisites

```bash
# Install Java 8
sudo apt install openjdk-8-jdk -y

# Install Python packages
pip3 install pyspark==3.4.0 kafka-python pandas streamlit

# Download and setup Kafka (standalone)
cd ~
wget https://archive.apache.org/dist/kafka/3.4.0/kafka_2.12-3.4.0.tgz
tar -xzf kafka_2.12-3.4.0.tgz
mv kafka_2.12-3.4.0 kafka
```

### 12.2 Download Dataset

```bash
cd ~/MiniProject3/data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
cp ml-1m/ratings.dat .
```

### 12.3 Start Kafka

```bash
cd ~/kafka

# Terminal 1 – Zookeeper
bin/zookeeper-server-start.sh config/zookeeper_custom.properties

# Terminal 2 – Kafka (after Zookeeper is ready)
bin/kafka-server-start.sh config/server_custom.properties

# Terminal 3 – Create topic (once)
bin/kafka-topics.sh --create --topic movie-ratings \
  --bootstrap-server localhost:9093 --partitions 2 --replication-factor 1
```

### 12.4 Run the System

```bash
# Terminal 4 – Train model
cd ~/MiniProject3
python3 src/model_training.py

# Terminal 5 – Streaming app
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \
  src/streaming_app.py

# Terminal 6 – Producer
python3 src/kafka_producer.py
```

### 12.5 View Results

```bash
# Watch JSON files update
watch -n 2 'ls -lh ~/MiniProject3/data/*.json'

# View analytics
cat ~/MiniProject3/data/dashboard_analytics.json | python3 -m json.tool | head -50

# Generate CSV
cd ~/MiniProject3/data
python3 -c "
import json, pandas as pd
with open('dashboard_analytics.json') as f:
    data = json.load(f)
pd.DataFrame(data['metrics']).to_csv('trending_items.csv', index=False)
"
```

---

## 13. Conclusion

This project successfully implements an **end-to-end real-time recommendation system** using Apache Spark and Kafka.

### Requirements Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ALS model on MovieLens 1M (1M+ records) | ✅ | `model_training.py` runs, RMSE=0.8765 |
| RMSE evaluation & tuning (if >1.5) | ✅ | RMSE<1.5, tuning skipped (code includes it) |
| Kafka producer with 2+ partitions, justified strategy | ✅ | Partition key = `item_id` – optimises aggregations |
| Spark Structured Streaming (30s window, 10s slide) | ✅ | `.groupBy(window(..., "30 seconds", "10 seconds"))` |
| Custom metric (`trending_score = interactions × avg_rating`) | ✅ | Present in code & output |
| Alert system (trending & spike) | ✅ | Log shows both alert types |
| Late data handling (watermark) | ✅ | `.withWatermark("event_time", "2 minutes")` |
| ML + streaming integration (top‑5 recommendations) | ✅ | `recommendForUserSubset` called per batch |
| <5 seconds latency (bonus) | ✅ | Measured 2–4 seconds |
| Results saved (JSON, CSV, logs, graphs) | ✅ | All files present |

### Key Achievements

- **Scalable batch training** on 1M records using Spark ALS.
- **Real‑time streaming** with Kafka and Spark Structured Streaming.
- **Dynamic recommendations** for active users with <5 second latency.
- **Trend detection** via custom `trending_score` metric.
- **Alert system** for trending items and activity spikes.
- **Complete logging** and JSON output for report generation.

The system demonstrates industry‑relevant skills in distributed data processing, real‑time analytics, and machine learning integration – essential for modern big data pipelines.

---

## 14. References

1. Apache Spark. (2024). Spark SQL, DataFrames and Datasets Guide. https://spark.apache.org/docs/latest/sql-programming-guide.html

2. Apache Kafka. (2024). Kafka Documentation. https://kafka.apache.org/documentation/

3. GroupLens Research. (2015). MovieLens 1M Dataset. https://grouplens.org/datasets/movielens/1m/

4. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. IEEE ICDM.

5. Zaharia, M., et al. (2016). Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM.

---

## Appendix:

---

**Screenshot 1 – Model Training Output**  

<img width="547" height="540" alt="Screenshot 2026-05-10 231743" src="https://github.com/user-attachments/assets/0aa92a4f-de40-4d16-87a0-4919d20defa4" />

---

**Screenshot 4 – Trending Items & Alerts**  


<img width="547" height="540" alt="Screenshot 2026-05-10 231743" src="https://github.com/user-attachments/assets/0430c72d-e0c2-4eab-9706-83180814e3bc" />

<img width="745" height="255" alt="Screenshot 2026-05-10 233522" src="https://github.com/user-attachments/assets/8512ae26-8679-417c-9c12-3bea6cb5211a" />

<img width="745" height="613" alt="Screenshot 2026-05-10 233531" src="https://github.com/user-attachments/assets/4a6c2009-4f75-4722-b66c-26d7986615da" />

---


**Screenshot 5 – Dynamic Recommendations**  

<img width="720" height="712" alt="Screenshot 2026-05-10 205002" src="https://github.com/user-attachments/assets/7dec1d83-087e-4064-9bfa-193cdb7b972f" />

---

**Screenshot 7 –  Graphs**  

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/26a19cd5-4c59-4bf2-881d-6d33cae624bb" />

<img width="1500" height="750" alt="image" src="https://github.com/user-attachments/assets/e787dbcb-ce58-4653-b46a-f40e02d3a559" />


<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/4925a9a4-e1cb-4a21-8ac8-edbc5c61f407" />


<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/b1cf97d2-9e7b-466b-94b4-71c8d0524886" />

---

**Report prepared by:** Bosy Ayman - Rihanna Nasr
**Student ID:** 202202076
**Date:** 2026-05-8

---
