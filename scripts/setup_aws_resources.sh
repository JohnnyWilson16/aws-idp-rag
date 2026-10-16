#!/usr/bin/env bash
# ==============================================================================
# AWS Serverless IDP + RAG Pipeline - Automation & Deployment Script
# ==============================================================================
set -e

STACK_NAME="aws-idp-rag-stack"
ENV="dev"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "=================================================================="
echo "⚡ AWS SERVERLESS IDP + RAG DEPLOYMENT (ZERO BEDROCK / FREE TIER)"
echo "=================================================================="

usage() {
    echo "Usage: $0 [build|deploy|delete|test-api|run-ui]"
    echo "  build     - Build SAM artifacts"
    echo "  deploy    - Deploy stack using AWS SAM (guided or auto)"
    echo "  delete    - Tear down all AWS resources to avoid charges"
    echo "  test-api  - Run test query against deployed API Gateway"
    echo "  run-ui    - Launch Streamlit frontend dashboard locally"
    exit 1
}

case "$1" in
    build)
        echo "--> Building SAM application..."
        sam build --use-container
        echo "Build complete."
        ;;

    deploy)
        echo "--> Deploying AWS SAM stack: ${STACK_NAME} in region ${REGION}..."
        sam deploy \
            --stack-name "${STACK_NAME}" \
            --region "${REGION}" \
            --resolve-s3 \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
            --parameter-overrides Environment="${ENV}"
        echo "Deployment complete! Fetching API Gateway endpoint..."
        aws cloudformation describe-stacks \
            --stack-name "${STACK_NAME}" \
            --region "${REGION}" \
            --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
            --output text
        ;;

    delete)
        echo "--> WARNING: Deleting stack ${STACK_NAME} in ${REGION} to prevent any charges..."
        # First empty S3 bucket
        BUCKET_NAME=$(aws cloudformation describe-stacks \
            --stack-name "${STACK_NAME}" \
            --region "${REGION}" \
            --query "Stacks[0].Outputs[?OutputKey=='DocumentStorageBucketName'].OutputValue" \
            --output text 2>/dev/null || echo "")

        if [ -n "$BUCKET_NAME" ] && [ "$BUCKET_NAME" != "None" ]; then
            echo "Emptying S3 bucket: s3://${BUCKET_NAME}..."
            aws s3 rm "s3://${BUCKET_NAME}" --recursive || true
        fi

        sam delete --stack-name "${STACK_NAME}" --region "${REGION}" --no-prompts
        echo "Teardown complete. All AWS resources deleted."
        ;;

    test-api)
        if [ -z "$2" ]; then
            echo "Error: Please provide API Gateway URL as second argument."
            echo "Example: $0 test-api https://abc.execute-api.us-east-1.amazonaws.com/dev"
            exit 1
        fi
        API_URL="$2"
        echo "--> Testing POST ${API_URL}/query..."
        curl -X POST "${API_URL}/query" \
            -H "Content-Type: application/json" \
            -d '{
                "question": "What is the liability cap in the contract?",
                "top_k": 3
            }' | jq .
        ;;

    run-ui)
        echo "--> Launching Streamlit Frontend..."
        streamlit run frontend/app.py
        ;;

    *)
        usage
        ;;
esac
