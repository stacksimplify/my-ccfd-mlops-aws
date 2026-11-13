-- MLOps 15 (BONUS): query the captured data with Athena. Replace <jobs-bucket> with your jobs bucket name.
CREATE DATABASE IF NOT EXISTS ccfd_monitoring;

CREATE EXTERNAL TABLE IF NOT EXISTS ccfd_monitoring.records (
  captureData struct<endpointInput:struct<data:string, encoding:string, mode:string, observedContentType:string>,
                     endpointOutput:struct<data:string, encoding:string, mode:string, observedContentType:string>>,
  eventMetadata struct<eventId:string, inferenceId:string, inferenceTime:string>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://<jobs-bucket>/data-capture/ccfd-prod/';
