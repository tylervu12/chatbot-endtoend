#!/usr/bin/env python3
import os

import aws_cdk as cdk

from infra.infra_stack import ChatbotStack


app = cdk.App()
ChatbotStack(app, "CompanyChatbotApiStack",
    # Deploy to current AWS CLI configured account/region
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'), 
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
    
    # Add stack description
    description="RAG-based chatbot API for company documents using Pinecone and OpenAI"
)

app.synth()
