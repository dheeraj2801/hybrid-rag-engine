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
