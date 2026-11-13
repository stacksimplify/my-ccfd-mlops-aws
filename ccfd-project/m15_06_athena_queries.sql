-- MLOps 15 (BONUS): 3 questions on the captured data (run after m15_06_athena_tables.sql).

-- 1. How many calls per replayed month?
SELECT split_part(eventMetadata.inferenceId, ':', 2) AS month, count(*) AS calls
FROM ccfd_monitoring.records GROUP BY 1 ORDER BY 1;

-- 2. How many FRAUD answers per month?
SELECT split_part(eventMetadata.inferenceId, ':', 2) AS month,
       count_if(captureData.endpointOutput.data LIKE '%"FRAUD"%') AS fraud_answers
FROM ccfd_monitoring.records GROUP BY 1 ORDER BY 1;

-- 3. The newest 10 calls
SELECT eventMetadata.inferenceTime, eventMetadata.inferenceId
FROM ccfd_monitoring.records ORDER BY eventMetadata.inferenceTime DESC LIMIT 10;
