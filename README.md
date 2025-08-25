# 🤖 Enterprise RAG Chatbot API - Complete Implementation Guide

## What This Is

This repository contains a **complete, production-ready chatbot system** that can answer questions about your company documents. The chatbot:

- 📚 **Reads your documents** (like employee handbooks, policies, FAQs)
- 🧠 **Understands questions** in natural language
- ✅ **Gives accurate answers** based only on your documents
- 🚀 **Runs in the cloud** on Amazon Web Services (AWS)
- 🔒 **Is secure** with API keys and rate limiting
- 💰 **Costs very little** to run (typically $5-20/month)

## What You'll Build

By following this guide, you'll create:

1. **A secure API endpoint** that your developers can use
2. **A document processing system** that automatically updates when you change documents
3. **A monitoring dashboard** to track usage and costs
4. **Rate limiting and security** to prevent abuse

## Prerequisites (What You Need)

### 1. Required Accounts
- **AWS Account** (Amazon Web Services) - [Sign up here](https://aws.amazon.com/)
- **OpenAI Account** - [Sign up here](https://platform.openai.com/)
- **Pinecone Account** - [Sign up here](https://www.pinecone.io/)

### 2. Required Software
- **Python 3.11 or newer** - [Download here](https://www.python.org/downloads/)
- **AWS CLI** - [Installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- **Node.js** (for AWS CDK) - [Download here](https://nodejs.org/)
- **Git** - [Download here](https://git-scm.com/)

### 3. Technical Knowledge Required
- Basic command line usage (copy/paste commands)
- Ability to edit text files
- Basic understanding of file/folder structure

**Don't worry!** This guide explains every step in detail.

---

## 🚀 Step-by-Step Implementation

### Step 1: Get the Code

1. **Download this repository** to your computer:
   ```bash
   git clone https://github.com/tylervu12/chatbot-endtoend
   cd chatbot-endtoend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Set Up Your API Keys

#### 2.1 Get OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in to your account
3. Navigate to "API Keys" in the left sidebar
4. Click "Create new secret key"
5. **Copy the key** (starts with `sk-`) - you'll need it later
6. **Add billing information** in your OpenAI account (required for API usage)

#### 2.2 Get Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign in to your account
3. Click on "API Keys" in the left sidebar
4. **Copy your API key** - you'll need it later

#### 2.3 Create Environment File
1. **Create a new file** called `.env` in the main folder
2. **Add your API keys** (replace with your actual keys):
   ```
   OPENAI_API_KEY=sk-your-openai-key-here
   PINECONE_API_KEY=your-pinecone-key-here
   ```
3. **Save the file**

> ⚠️ **Important**: Never share your `.env` file or commit it to Git!

### Step 3: Prepare Your Documents

#### 3.1 Add Your Company Documents
1. **Open the `data/raw/` folder**
2. **Replace the example files** with your own documents:
   - Employee handbook
   - Company policies  
   - FAQ documents
   - Procedures
   - Any other text-based company documents

#### 3.2 Document Requirements
- **File format**: `.txt` files only
- **Content**: Plain text (copy from Word/Google Docs and save as .txt)

**Example file structure:**
```
data/raw/
├── employee_handbook.txt
├── company_policies.txt
├── benefits_guide.txt
└── vacation_policy.txt
```

### Step 4: Process Your Documents

#### 4.1 Run Document Processing
This step converts your documents into a format the AI can understand:

```bash
python src/local_processing/process_documents.py
```

**What this does:**
- Reads all `.txt` files in `data/raw/`
- Breaks them into smaller chunks
- Adds document names to each chunk for better context
- Saves processed data to `data/processed/`


### Step 5: Upload to Pinecone (Vector Database)

#### 5.1 Create Pinecone Index
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Click "Create Index"
3. **Settings**:
   - Name: `chatbot-rag-index`
   - Dimensions: `1536`
   - Metric: `cosine`
   - Cloud: `AWS`
   - Region: `us-east-1`
4. Click "Create Index"

#### 5.2 Upload Your Documents
```bash
python src/local_processing/upload_to_pinecone.py
```

**What this does:**
- Converts your document chunks into mathematical vectors
- Uploads them to Pinecone for fast searching
- Verifies the upload was successful

### Step 6: Set Up AWS Infrastructure

#### 6.1 Configure AWS CLI
1. **Install AWS CLI** (if not already done)
2. **Configure it** with your AWS credentials:
   ```bash
   aws configure
   ```
3. **Enter your**:
   - AWS Access Key ID
   - AWS Secret Access Key  
   - Default region: `us-east-1`
   - Default output format: `json`

> 💡 **Need AWS credentials?** Go to AWS Console → IAM → Users → Create User → Attach policies

#### 6.2 Install AWS CDK
```bash
npm install -g aws-cdk
```

#### 6.3 Bootstrap CDK (First time only)
```bash
cd infra
cdk bootstrap
```

### Step 7: Store API Keys in AWS

Before deploying, you need to securely store your API keys in AWS:

#### 7.1 Store OpenAI Key
```bash
aws secretsmanager create-secret \
    --name "chatbot/openai-api-key" \
    --description "OpenAI API Key for RAG chatbot" \
    --secret-string "sk-your-openai-key-here"
```

#### 7.2 Store Pinecone Key  
```bash
aws secretsmanager create-secret \
    --name "chatbot/pinecone-api-key" \
    --description "Pinecone API Key for vector storage" \
    --secret-string "your-pinecone-key-here"
```

### Step 8: Deploy to AWS

#### 8.1 Build Lambda Dependencies
```bash
cd ../lambda-layer
chmod +x build.sh
./build.sh
```

#### 8.2 Deploy the Infrastructure
```bash
cd ../infra
cdk deploy
```

**This creates:**
- Lambda function (your chatbot brain)
- API Gateway (your chatbot's web address)
- Security settings
- Monitoring


#### 8.3 Get Your API Information
After deployment, you'll see outputs like:
```
CompanyChatbotApiStack.APIEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod
CompanyChatbotApiStack.APIKeyId = abc123def456
```

**Save these values!**

### Step 9: Get Your API Key

```bash
aws apigateway get-api-key --api-key abc123def456 --include-value
```

**Copy the `value` field** - this is your API key for making requests.

---

## 🧪 Testing Your Chatbot

### Test with curl (Command Line)
```bash
curl -X POST "https://your-api-endpoint.amazonaws.com/prod/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key-here" \
  -d '{"message": "What is our vacation policy?"}'
```

### Test with Postman (GUI Tool)
1. **Download Postman** (free API testing tool)
2. **Create new POST request**
3. **URL**: Your API endpoint + `/chat`
4. **Headers**:
   - `Content-Type: application/json`
   - `x-api-key: your-api-key-here`
5. **Body** (JSON):
   ```json
   {"message": "What are our company benefits?"}
   ```

### Expected Response
```json
{
  "answer": "According to our benefits guide, we offer...",
  "sources": ["benefits_guide.txt"],
  "chunks_used": 2,
  "processing_time_ms": 1500
}
```

---

## 📊 Monitoring & Maintenance

### View Logs
1. Go to **AWS Console** → **CloudWatch** → **Log Groups**
2. Find `/aws/lambda/chatbot-rag-function`
3. View recent requests and any errors

### Monitor Costs
1. **AWS Console** → **Billing & Cost Management** → **Bills**
2. Look for charges from:
   - Lambda (compute time)
   - API Gateway (requests)
   - CloudWatch (logging)

### Update Documents
To add or change documents:

1. **Update files** in `data/raw/`
2. **Reprocess documents**:
   ```bash
   python src/local_processing/process_documents.py
   ```
3. **Upload to Pinecone**:
   ```bash
   python src/local_processing/upload_to_pinecone.py
   ```

The system automatically detects changed files and only processes what's new!

---

## 💰 Expected Costs

### Monthly Estimates (for typical usage)
- **OpenAI API**: $5-15 (depends on questions asked)
- **Pinecone**: $0-10 (free tier covers most small businesses)
- **AWS Services**: $2-5 (Lambda + API Gateway + storage)

**Total: $7-30/month** for a small business

### Cost Optimization Tips
- Set up **AWS Budget alerts** for unexpected charges
- Monitor **OpenAI usage** in their dashboard
- Use **CloudWatch dashboards** to track API calls

---

## 🔧 Customization Options

### Adjust AI Behavior
Edit `src/functions/chat/handler.py` lines 143-153 to change:
- Response tone (formal/casual)
- Answer length
- Citation requirements

### Change Rate Limits
Edit `infra/infra/infra_stack.py` lines 114-121:
- Requests per minute
- Monthly quotas
- Burst limits

### Add New Document Types
The system currently supports `.txt` files. To add others:
1. Modify `src/local_processing/process_documents.py`
2. Add file format conversion logic

---

## 🆘 Troubleshooting

### Common Issues

#### "Permission Denied" Errors
- **Check AWS CLI configuration**: `aws sts get-caller-identity`
- **Verify IAM permissions** for your AWS user

#### "API Key Invalid" Errors  
- **Verify API keys** in your `.env` file
- **Check OpenAI billing** is set up
- **Confirm Pinecone key** is correct

#### "No chunks found" Responses
- **Check document processing** completed successfully
- **Verify Pinecone upload** worked
- **Try more specific questions**

#### High Costs
- **Check OpenAI usage** dashboard
- **Review CloudWatch metrics** for unexpected traffic
- **Verify rate limiting** is working

### Getting Help

1. **Check the logs** in CloudWatch first
2. **Review error messages** carefully
3. **Test with simple questions** to isolate issues
4. **Verify each step** was completed successfully

### Debug Mode
Add this to your `.env` file for more detailed logging:
```
LOG_LEVEL=DEBUG
```

---

## 🔒 Security Best Practices

### Protect Your API Key
- **Never hardcode** API keys in your application
- **Use environment variables** or secure key management
- **Rotate keys regularly** (every 90 days)
- **Monitor usage** for unusual activity

### Access Control
- **Limit API key distribution** to authorized users only
- **Set up monitoring alerts** for high usage
- **Use different keys** for development vs production

### Data Privacy
- **Review your documents** before uploading
- **Remove sensitive information** (SSNs, personal data)
- **Consider data classification** policies

---

## 🎯 Next Steps

### For Developers
Once your chatbot is working, developers can:
- **Build web interfaces** using the API
- **Create mobile apps** that connect to your chatbot
- **Integrate with existing systems** (Slack, Teams, etc.)

### API Documentation for Developers
```
Endpoint: POST /chat
Headers: 
  - Content-Type: application/json
  - x-api-key: {your-api-key}

Request Body:
{
  "message": "Your question here"
}

Response:
{
  "answer": "AI-generated response",
  "sources": ["document1.txt", "document2.txt"],
  "chunks_used": 3,
  "processing_time_ms": 1200
}

Rate Limits: 100 requests/minute
```

### Scaling Considerations
As your usage grows:
- **Monitor performance** metrics
- **Consider increasing** Lambda memory/timeout
- **Review Pinecone** index size limits
- **Implement caching** for common questions

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- **Monthly**: Review costs and usage
- **Quarterly**: Rotate API keys
- **As needed**: Update documents
- **Annual**: Review and update dependencies

### System Updates
This repository may receive updates. To apply them:
1. **Backup your customizations**
2. **Pull latest changes**: `git pull`
3. **Redeploy**: `cdk deploy`

---

## 🎉 Congratulations!

You now have a **production-ready, enterprise-grade chatbot** that:
- ✅ Answers questions about your company documents
- ✅ Runs securely in the cloud
- ✅ Scales automatically with usage
- ✅ Costs only a few dollars per month
- ✅ Can be integrated into any application

Your chatbot is ready to help employees, customers, or anyone who needs information from your documents!

