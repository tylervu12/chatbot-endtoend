from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    aws_secretsmanager as secretsmanager,
    aws_iam as iam,
    aws_logs as logs,
    CfnOutput,
    RemovalPolicy
)
from constructs import Construct

class ChatbotStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Secrets Manager for API Keys
        openai_secret = secretsmanager.Secret(
            self, "OpenAIAPIKey",
            description="OpenAI API Key for RAG chatbot",
            secret_name="chatbot/openai-api-key"
        )

        pinecone_secret = secretsmanager.Secret(
            self, "PineconeAPIKey", 
            description="Pinecone API Key for vector storage",
            secret_name="chatbot/pinecone-api-key"
        )

        # 2. IAM Role for Lambda Function
        lambda_role = iam.Role(
            self, "ChatLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # Grant access to secrets
        openai_secret.grant_read(lambda_role)
        pinecone_secret.grant_read(lambda_role)

        # 3. CloudWatch Log Group
        log_group = logs.LogGroup(
            self, "ChatLambdaLogGroup",
            log_group_name="/aws/lambda/chatbot-rag-function",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY
        )

        # 4. Lambda Layers
        # Custom dependencies layer
        dependencies_layer = _lambda.LayerVersion(
            self, "ChatbotDependenciesLayer",
            code=_lambda.Code.from_asset("../lambda-layer/layer.zip"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
            compatible_architectures=[_lambda.Architecture.ARM_64],
            description="Dependencies layer for chatbot (OpenAI, Pinecone, Pydantic)"
        )
        
        # Note: Removed AWS NumPy layer due to permission issues
        # Our custom layer should handle all required dependencies

        # 5. Lambda Function
        chat_function = _lambda.Function(
            self, "ChatFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            architecture=_lambda.Architecture.ARM_64,  # Match layer architecture
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("../src/functions/chat"),
            role=lambda_role,
            function_name="chatbot-rag-function",
            timeout=Duration.seconds(300),
            memory_size=1024,
            layers=[dependencies_layer],
            environment={
                "OPENAI_API_KEY_SECRET_ARN": openai_secret.secret_arn,
                "PINECONE_API_KEY_SECRET_ARN": pinecone_secret.secret_arn,
                "PINECONE_INDEX_NAME": "chatbot-rag-index",
                "RAG_TOP_K": "3",
                "RAG_SCORE_THRESHOLD": "0.5",
                "OPENAI_LLM_MODEL": "gpt-4o",
                "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
                "CACHE_BUSTER": "v3"
            },
            log_group=log_group
        )

        # 6. API Gateway
        api = apigateway.RestApi(
            self, "ChatbotAPI",
            rest_api_name="Chatbot RAG API",
            description="RAG-based chatbot API for company documents",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "x-api-key"]
            )
        )

        # 7. API Key and Usage Plan
        api_key = api.add_api_key(
            "ChatbotAPIKey",
            api_key_name="chatbot-api-key",
            description="API key for chatbot access"
        )

        usage_plan = api.add_usage_plan(
            "ChatbotUsagePlan",
            name="chatbot-usage-plan",
            description="Usage plan for chatbot API",
            throttle=apigateway.ThrottleSettings(
                rate_limit=100,  # 100 requests per minute
                burst_limit=200
            ),
            quota=apigateway.QuotaSettings(
                limit=10000,  # 10,000 requests per month
                period=apigateway.Period.MONTH
            )
        )

        usage_plan.add_api_key(api_key)

        # 8. Lambda Integration
        chat_integration = apigateway.LambdaIntegration(
            chat_function,
            request_templates={
                "application/json": '{"statusCode": "200"}'
            }
        )

        # 9. API Resource and Method
        chat_resource = api.root.add_resource("chat")
        chat_method = chat_resource.add_method(
            "POST",
            chat_integration,
            api_key_required=True,
            request_validator=apigateway.RequestValidator(
                self, "ChatRequestValidator",
                rest_api=api,
                validate_request_body=True,
                request_validator_name="chat-request-validator"
            ),
            request_models={
                "application/json": apigateway.Model(
                    self, "ChatRequestModel",
                    rest_api=api,
                    content_type="application/json",
                    model_name="ChatRequest",
                    schema=apigateway.JsonSchema(
                        type=apigateway.JsonSchemaType.OBJECT,
                        properties={
                            "message": apigateway.JsonSchema(
                                type=apigateway.JsonSchemaType.STRING,
                                min_length=1,
                                max_length=1000
                            )
                        },
                        required=["message"]
                    )
                )
            }
        )

        # Add usage plan to API stage
        usage_plan.add_api_stage(
            stage=api.deployment_stage
        )

        # 10. CloudWatch Dashboard (optional)
        # Could add custom metrics dashboard here

        # 11. Outputs
        CfnOutput(
            self, "APIEndpoint",
            value=api.url,
            description="API Gateway endpoint URL"
        )

        CfnOutput(
            self, "APIKeyId", 
            value=api_key.key_id,
            description="API Key ID (use 'aws apigateway get-api-key' to retrieve the key value)"
        )

        CfnOutput(
            self, "LambdaFunctionName",
            value=chat_function.function_name,
            description="Lambda function name"
        )

        CfnOutput(
            self, "OpenAISecretArn",
            value=openai_secret.secret_arn,
            description="OpenAI API Key Secret ARN"
        )

        CfnOutput(
            self, "PineconeSecretArn", 
            value=pinecone_secret.secret_arn,
            description="Pinecone API Key Secret ARN"
        )
