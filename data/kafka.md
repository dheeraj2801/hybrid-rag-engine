# Apache Kafka — Practical Notes

Apache Kafka is a distributed commit log and event streaming platform designed for high-throughput, low-latency message delivery. It is commonly used for data pipelines, stream processing, and event-driven architectures.

Key concepts

- Topic: a named feed to which records are published. Topics are partitioned for parallelism.
- Partition: an ordered, immutable sequence of messages; the unit of parallelism and ordering.
- Broker: a Kafka server that stores partitions and serves clients.
- Producer / Consumer: clients that write to and read from topics.
- Offset: a position in a partition that uniquely identifies a record.
- Consumer Group: a set of consumers that share work by consuming different partitions.

Core usage patterns

- Pub/Sub: broadcast events to multiple subscribers using consumer groups or separate topics.
- Event Sourcing: persist state transitions as an immutable log and rebuild state by replaying events.
- Stream Processing: apply transformations with frameworks like Kafka Streams, ksqlDB, or Flink.

Example (Python, synchronous producer using confluent-kafka)

```python
from confluent_kafka import Producer

conf = {'bootstrap.servers': 'localhost:9092'}
p = Producer(conf)

def delivery_report(err, msg):
	if err is not None:
		print('Delivery failed:', err)
	else:
		print('Delivered to', msg.topic(), msg.partition())

p.produce('my-topic', key='user:1', value='{"event":"login"}', callback=delivery_report)
p.flush()
```

Async consumers

- Use `aiokafka` for async consumers in Python when integrating with `asyncio`-based servers (FastAPI, aiohttp).

Operational concerns

- Partitions: choose partition count based on expected throughput and consumer parallelism.
- Replication: set replication factor >= 2 for fault tolerance.
- Retention: configure retention.ms or size-based retention for storage management.
- Compaction: enable log compaction for changelog/event-sourcing topics.

Monitoring & tuning

- Monitor consumer lag, under-replicated partitions, and ISR size.
- Tune `num.io.threads`, `num.network.threads`, and JVM heap for brokers.
- Use tools: `kafka-consumer-groups.sh`, Kafka Manager, Prometheus + JMX exporter.

When to use Kafka

- High-throughput event ingestion (millions of events/second at scale).
- Durable ordered logs for replay and event sourcing.
- Loose coupling across services where asynchronous integration and backpressure handling are required.

Alternatives and complements: RabbitMQ, Pulsar, Kinesis, and lightweight message buses for simpler needs.

References

- Kafka docs: https://kafka.apache.org
- aiokafka, confluent-kafka Python clients

When evaluating whether to use Kafka, consider throughput, ordering, and durability requirements. Kafka is appropriate when you need a durable, partitioned log with replayability and high throughput. For lightweight pub/sub or low-throughput eventing, managed message queues or simple HTTP-based webhooks may be preferable. Architect for scale by planning partition counts, horizontal consumer groups, and proper monitoring of lag and broker resource usage.

Inspect consumer groups and offsets:

```sh
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group my-group
```

Check broker log location and tail errors (example path depends on installation):

```sh
tail -f /var/log/kafka/server.log
```

Partition: an ordered, immutable sequence of records within a topic; the primary unit of parallelism.

Offset: a numerical index for a message within a partition.

Replication factor: number of brokers storing copies of a partition for fault tolerance.

Log compaction: retention policy that keeps only the latest value for each message key.

`NotLeaderForPartition`: producer attempted to write to a broker that is not the leader; reconcile by refreshing metadata, checking leader election, and ensuring brokers are healthy.

`OffsetOutOfRange`: consumer requested an offset that no longer exists, often after retention; handle by resetting offsets (earliest/latest) or using committed checkpoints.

Broker OOM / disk full: monitor disk usage and JVM memory; add retention/compaction or increase storage.

Improving both throughput and durability requires tuning several levers: increase partitions to scale throughput, set replication factor >1 for durability, tune producer batching (`linger.ms`, `batch.size`) and acks (e.g., `acks=all`) for stronger durability guarantees, and monitor via JMX/Prometheus. Design for operational recovery (automated broker replacement, partition reassignment) and test rebalancing effects on consumer lag in staging.
