#!/usr/bin/env python3
"""
Test script to verify ELK stack integration
"""

import requests
import time
import json

def test_elasticsearch_connection():
    """Test if Elasticsearch is running and accessible"""
    try:
        response = requests.get("http://localhost:9200")
        if response.status_code == 200:
            print("✅ Elasticsearch is running")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Elasticsearch returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Elasticsearch on localhost:9200")
        print("Make sure the ELK stack is running with: cd elk && docker-compose up -d")
        return False

def test_kibana_connection():
    """Test if Kibana is running and accessible"""
    try:
        response = requests.get("http://localhost:5601")
        if response.status_code == 200:
            print("✅ Kibana is running")
            return True
        else:
            print(f"❌ Kibana returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Kibana on localhost:5601")
        return False

def check_indexes():
    """Check if the expected indexes exist in Elasticsearch"""
    try:
        # Check for data_preprocessing index
        response = requests.get("http://localhost:9200/data_preprocessing_logs")
        if response.status_code == 200:
            print("✅ data_preprocessing_logs index exists")
        else:
            print("❌ data_preprocessing_logs index not found")
        
        # Check for model_training index
        response = requests.get("http://localhost:9200/model_training_logs")
        if response.status_code == 200:
            print("✅ model_training_logs index exists")
        else:
            print("❌ model_training_logs index not found")
        
        # Check for model_deployment index
        response = requests.get("http://localhost:9200/model_deployment_logs")
        if response.status_code == 200:
            print("✅ model_deployment_logs index exists")
        else:
            print("❌ model_deployment_logs index not found")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Elasticsearch")

def get_index_stats():
    """Get statistics about the indexes"""
    try:
        response = requests.get("http://localhost:9200/_cat/indices?v")
        if response.status_code == 200:
            print("\n📊 Index Statistics:")
            print(response.text)
        else:
            print("❌ Could not get index statistics")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Elasticsearch")

def main():
    print("🔍 Testing ELK Stack Integration")
    print("=" * 50)
    
    # Test connections
    es_running = test_elasticsearch_connection()
    kibana_running = test_kibana_connection()
    
    if es_running:
        print("\n📋 Checking Indexes:")
        check_indexes()
        
        print("\n📊 Getting Index Statistics:")
        get_index_stats()
    
    print("\n" + "=" * 50)
    if es_running and kibana_running:
        print("✅ ELK stack is running properly!")
        print("🌐 Access Kibana at: http://localhost:5601")
        print("🔍 Access Elasticsearch at: http://localhost:9200")
    else:
        print("❌ ELK stack is not running properly")
        print("💡 To start the ELK stack:")
        print("   cd elk")
        print("   docker-compose up -d")

if __name__ == "__main__":
    main() 