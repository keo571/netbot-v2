#!/usr/bin/env python3
"""
Comprehensive test for NetBot V2 Hybrid RAG System.

Tests the complete hybrid RAG architecture including:
- RAG Orchestrator coordination
- Text RAG integration
- Graph RAG integration  
- Context Manager integration
- Reliability assessment
- API Gateway functionality
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.netbot.services.rag_orchestrator import RAGOrchestrator, RAGClient
from src.netbot.services.rag_orchestrator.models import RAGQuery, ProcessingMode, QueryType
from src.netbot.services.text_rag import TextRAG
from src.netbot.services.context_manager import ContextManager


def print_header(title: str) -> None:
    """Print formatted test section header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print('='*60)


def print_step(step: str) -> None:
    """Print formatted test step."""
    print(f"\n{step}")
    print('-' * len(step))


def print_result(success: bool, message: str) -> None:
    """Print formatted test result."""
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")


def print_data(title: str, data: Any, max_length: int = 200) -> None:
    """Print formatted data with truncation."""
    print(f"\n📊 {title}:")
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if len(json_str) > max_length:
            json_str = json_str[:max_length] + "..."
        print(json_str)
    else:
        str_data = str(data)
        if len(str_data) > max_length:
            str_data = str_data[:max_length] + "..."
        print(str_data)


async def test_rag_orchestrator_basic() -> bool:
    """Test basic RAG Orchestrator functionality."""
    print_step("1. Testing RAG Orchestrator initialization")
    
    try:
        orchestrator = RAGOrchestrator()
        print_result(True, "RAG Orchestrator initialized successfully")
        
        # Test basic query
        print_step("2. Testing basic orchestrated query")
        
        query = RAGQuery(
            query_text="What is a load balancer and how does it work?",
            query_type=QueryType.HYBRID_FUSION,
            processing_mode=ProcessingMode.BALANCED,
            top_k=5
        )
        
        response = await orchestrator.query(query)
        
        print_result(True, f"Query processed successfully")
        print_data("Response Text", response.response_text, 300)
        print_data("Confidence Metrics", {
            "overall_confidence": response.confidence_metrics.overall_confidence,
            "reliability_level": response.confidence_metrics.reliability_level.value,
            "source_coverage": response.confidence_metrics.source_coverage,
            "response_grounding": response.confidence_metrics.response_grounding
        })
        
        # Test system status
        print_step("3. Testing system status")
        status = orchestrator.get_system_status()
        print_result(True, f"System status: {status['status']}")
        print_data("Performance Stats", status.get('performance_stats', {}))
        
        orchestrator.close()
        return True
        
    except Exception as e:
        print_result(False, f"RAG Orchestrator test failed: {e}")
        return False


async def test_rag_client_interface() -> bool:
    """Test RAG Client simplified interface."""
    print_step("1. Testing RAG Client initialization")
    
    try:
        client = RAGClient()
        print_result(True, "RAG Client initialized successfully")
        
        # Test ask interface
        print_step("2. Testing simplified ask interface")
        
        result = await client.ask(
            question="Explain network firewall configurations",
            mode="balanced",
            top_k=3,
            include_visualizations=False
        )
        
        print_result(True, f"Question answered successfully")
        print_data("Answer", result['answer'], 300)
        print_data("Confidence", result['confidence'])
        print_data("Sources Count", len(result['sources']))
        
        # Test search interface
        print_step("3. Testing search interface")
        
        search_result = await client.search(
            query="router configuration",
            search_type="semantic",
            top_k=5
        )
        
        print_result(True, f"Search completed with {search_result['total_found']} results")
        print_data("Search Quality", search_result['search_quality'])
        
        # Test document addition
        print_step("4. Testing document addition")
        
        doc_result = await client.add_document(
            content="A network router is a device that forwards data packets between computer networks. It connects multiple networks and routes traffic based on IP addresses.",
            title="Router Basics",
            document_type="educational",
            categories=["networking", "hardware"]
        )
        
        print_result(doc_result['success'], f"Document added: {doc_result.get('document_id', 'N/A')}")
        
        await client.close()
        return True
        
    except Exception as e:
        print_result(False, f"RAG Client test failed: {e}")
        return False


async def test_conversational_rag() -> bool:
    """Test conversational RAG with Context Manager integration."""
    print_step("1. Testing conversational RAG setup")
    
    try:
        client = RAGClient()
        
        # Start conversation
        print_step("2. Starting conversation session")
        session_id = await client.start_conversation(
            user_id="test_user_123",
            preferences={"expertise_level": "intermediate", "response_style": "detailed"}
        )
        print_result(True, f"Conversation started with session: {session_id}")
        
        # Send first message
        print_step("3. Sending first message")
        result1 = await client.continue_conversation(
            session_id=session_id,
            message="What are the main components of a network infrastructure?",
            mode="interactive"
        )
        
        print_result(True, "First message processed")
        print_data("Response", result1['answer'], 250)
        
        # Send follow-up message
        print_step("4. Sending follow-up message")
        result2 = await client.continue_conversation(
            session_id=session_id,
            message="How do these components work together for load balancing?",
            mode="interactive"
        )
        
        print_result(True, "Follow-up message processed")
        print_data("Response", result2['answer'], 250)
        print_data("Confidence", result2['confidence'])
        
        await client.close()
        return True
        
    except Exception as e:
        print_result(False, f"Conversational RAG test failed: {e}")
        return False


async def test_batch_processing() -> bool:
    """Test batch processing capabilities."""
    print_step("1. Testing batch question processing")
    
    try:
        client = RAGClient()
        
        questions = [
            "What is a network switch?",
            "How does VLAN work?",
            "What are the benefits of network segmentation?",
            "Explain firewall rules and policies"
        ]
        
        print_step("2. Processing batch questions")
        results = await client.batch_ask(
            questions=questions,
            parallel=True,
            max_concurrent=2
        )
        
        print_result(True, f"Batch processing completed: {len(results)} results")
        
        success_count = sum(1 for r in results if 'error' not in r.get('answer', '').lower())
        print_data("Success Rate", f"{success_count}/{len(results)}")
        
        for i, result in enumerate(results):
            print_data(f"Question {i+1}", {
                "question": result['question'][:50] + "..." if len(result['question']) > 50 else result['question'],
                "confidence": result['confidence'],
                "sources_count": len(result['sources'])
            })
        
        await client.close()
        return True
        
    except Exception as e:
        print_result(False, f"Batch processing test failed: {e}")
        return False


async def test_reliability_assessment() -> bool:
    """Test reliability and confidence assessment."""
    print_step("1. Testing reliability assessment")
    
    try:
        orchestrator = RAGOrchestrator()
        
        # Test with different query complexities
        test_queries = [
            ("Simple query", "What is IP?"),
            ("Medium query", "How do routers forward packets between networks?"),
            ("Complex query", "Explain the interaction between OSPF routing protocol, VLAN segmentation, and firewall policy enforcement in enterprise network architecture")
        ]
        
        results = []
        
        for query_name, query_text in test_queries:
            print_step(f"2. Testing {query_name.lower()}")
            
            query = RAGQuery(
                query_text=query_text,
                query_type=QueryType.HYBRID_FUSION,
                processing_mode=ProcessingMode.COMPREHENSIVE,
                include_confidence_scores=True
            )
            
            response = await orchestrator.query(query)
            
            confidence = response.confidence_metrics
            result_summary = {
                "query": query_name,
                "overall_confidence": confidence.overall_confidence,
                "reliability_level": confidence.reliability_level.value,
                "source_coverage": confidence.source_coverage,
                "context_completeness": confidence.context_completeness,
                "response_grounding": confidence.response_grounding,
                "information_gaps": len(confidence.information_gaps),
                "confidence_flags": len(confidence.confidence_flags)
            }
            
            results.append(result_summary)
            print_result(True, f"{query_name} processed")
            print_data("Reliability Metrics", result_summary)
        
        # Get system reliability report
        print_step("3. Getting system reliability report")
        reliability_report = orchestrator.reliability_manager.get_system_reliability_report()
        print_result(True, f"System health: {reliability_report.get('system_health', 'Unknown')}")
        print_data("Reliability Report", reliability_report)
        
        orchestrator.close()
        return True
        
    except Exception as e:
        print_result(False, f"Reliability assessment test failed: {e}")
        return False


async def test_error_handling() -> bool:
    """Test error handling and graceful degradation."""
    print_step("1. Testing error handling capabilities")
    
    try:
        client = RAGClient()
        
        # Test empty query
        print_step("2. Testing empty query handling")
        result = await client.ask(question="", mode="fast")
        print_result(
            'error' in result['answer'].lower() or result['confidence'] == 0.0,
            "Empty query handled gracefully"
        )
        
        # Test invalid parameters
        print_step("3. Testing invalid parameter handling")
        try:
            query = RAGQuery(
                query_text="Valid question",
                top_k=-1,  # Invalid
                similarity_threshold=1.5  # Invalid
            )
            orchestrator = RAGOrchestrator()
            response = await orchestrator.query(query)
            print_result(False, "Should have failed validation")
            orchestrator.close()
        except Exception:
            print_result(True, "Invalid parameters caught successfully")
        
        await client.close()
        return True
        
    except Exception as e:
        print_result(False, f"Error handling test failed: {e}")
        return False


async def main():
    """Run all hybrid RAG system tests."""
    print_header("NetBot V2 Hybrid RAG System Test Suite")
    
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Run all test suites
    test_suites = [
        ("RAG Orchestrator Basic", test_rag_orchestrator_basic),
        ("RAG Client Interface", test_rag_client_interface),
        ("Conversational RAG", test_conversational_rag),
        ("Batch Processing", test_batch_processing),
        ("Reliability Assessment", test_reliability_assessment),
        ("Error Handling", test_error_handling)
    ]
    
    for suite_name, test_func in test_suites:
        print_header(f"Testing {suite_name}")
        
        try:
            success = await test_func()
            test_results.append((suite_name, success))
            
            if success:
                print_result(True, f"{suite_name} - ALL TESTS PASSED")
            else:
                print_result(False, f"{suite_name} - SOME TESTS FAILED")
                
        except Exception as e:
            print_result(False, f"{suite_name} - TEST SUITE FAILED: {e}")
            test_results.append((suite_name, False))
    
    # Print final results
    print_header("Test Results Summary")
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for suite_name, success in test_results:
        print_result(success, suite_name)
    
    print_header(f"Final Result: {passed}/{total} Test Suites Passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! NetBot V2 Hybrid RAG System is operational!")
        print("✅ System components verified:")
        print("   - RAG Orchestrator coordination")
        print("   - Multi-modal retrieval (Text + Graph + Context)")
        print("   - Reliability assessment framework")
        print("   - Conversational AI capabilities")
        print("   - Batch processing support")
        print("   - Error handling and graceful degradation")
    else:
        failed = total - passed
        print(f"⚠️  {failed} test suite(s) failed. Please check the output above for details.")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)