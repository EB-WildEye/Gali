Technical questions:

1.  ChatLambda --> DynamoDB
maybe change to context json instead of DB at all? summeries once at X tokens and then just send the summary and last N messages? 

2. lambda - limitations:
--> by AWS agent:
While AWS Lambda offers many benefits, there are some limitations to consider:
Cold starts - Functions may experience latency when invoked after being idle, as Lambda needs to initialize the execution environment.
Execution time limits - Lambda functions have a maximum execution timeout of 15 minutes, making them unsuitable for long-running processes.
Resource constraints - Functions are limited in memory (up to 10 GB) and ephemeral storage (up to 10 GB in /tmp directory).
Debugging complexity - Troubleshooting serverless applications can be more challenging compared to traditional server-based applications.
Vendor lock-in - Applications built specifically for Lambda may require significant refactoring to migrate to other platforms.
Limited regional availability - Some Lambda features may not be available in all AWS regions.
Stateless nature - Lambda functions are stateless, requiring external services like Amazon S3 or DynamoDB for persistent data storage.
--> what are the solutions? is it worth it?

3. adding Summarization to the chat life cycle - minimal cost, minimal tokens, minimal latency

4. consider latency optimaztion with S3 Express One Zone, maybe not nessecary for now, but good to know

5. DynamoDB or json for chat context without history chat ? 

6. App factory - what do you think about it?

