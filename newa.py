from flask import Flask, request, jsonify, render_template, session
# from chatbot.flow_bot import FlowBasedChatbot, get_bot_response, get_enhanced_response
from chatbot.flow_botgr import FlowBasedChatbot
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'pu-chatbot-secret-key-change-in-production'

# Global dictionary to store chatbot instances for each session
chatbot_sessions = {}

def get_or_create_chatbot(session_id):
    """Get existing chatbot instance or create new one for session"""
    if session_id not in chatbot_sessions:
        chatbot_sessions[session_id] = FlowBasedChatbot()
    return chatbot_sessions[session_id]

@app.route("/")
def index():
    """Main page"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Main page"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index1.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    """Main chatbot endpoint"""
    data = request.get_json()
    user_message = data.get("message", "")
    # Get or create session ID
    session_id = session.get('session_id', str(uuid.uuid4()))
    if 'session_id' not in session:
        session['session_id'] = session_id
    try:
        # Get flow-based chatbot instance for this session
        chatbot = get_or_create_chatbot(session_id)
        # Process message with flow-based chatbot
        bot_data = chatbot.process_message(user_message)
        return jsonify({
            "reply": bot_data["response"],
            "recommendations": bot_data["recommendations"],
            "intent": bot_data["intent"],
            "confidence": bot_data["confidence"],
            "current_stage": bot_data.get("current_stage", "general"),  # fallback
            "current_intent": bot_data["intent"],  # new
            "conversation_summary": bot_data["conversation_summary"],
            "session_id": session_id
        })
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify({
            "reply": "I apologize, but I'm experiencing some technical difficulties. Please try again.",
            "recommendations": ["Hello", "Show me courses", "Tell me about admissions", "What is PU known for?"],
            "intent": "error",
            "confidence": 0.0,
            "current_stage": "error",
            "current_intent": None,
            "session_id": session_id
        })
    
@app.route("/get_conversation_summary", methods=["GET"])
def get_conversation_summary():
    """Get conversation summary for current session"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in chatbot_sessions:
        return jsonify({"error": "No active conversation found"}), 404
    
    try:
        chatbot = chatbot_sessions[session_id]
        summary = chatbot.get_conversation_summary()
        
        return jsonify({
            "summary": summary,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error getting summary: {e}")
        return jsonify({"error": "Could not retrieve conversation summary"}), 500

@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    """Reset conversation for current session"""
    session_id = session.get('session_id')
    
    try:
        if session_id and session_id in chatbot_sessions:
            chatbot_sessions[session_id].reset_conversation()
            message = "Conversation reset successfully"
        else:
            # Create new session if none exists
            if not session_id:
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
            chatbot_sessions[session_id] = FlowBasedChatbot()
            message = "New conversation started"
        
        return jsonify({
            "message": message,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error resetting conversation: {e}")
        return jsonify({"error": "Could not reset conversation"}), 500

@app.route("/get_flow_status", methods=["GET"])
def get_flow_status():
    """Get current flow status (safe access)"""
    session_id = session.get('session_id')
    if not session_id or session_id not in chatbot_sessions:
        return jsonify({
            "current_stage": "initial",
            "current_intent": None,
            "course_interest": None,
            "session_id": session_id
        })
    try:
        chatbot = chatbot_sessions[session_id]
        flow = chatbot.conversation_flow
        return jsonify({
            "current_stage": flow.get("current_stage", "general"),
            "current_intent": flow.get("current_intent"),
            "previous_intent": flow.get("previous_intent"),
            "course_interest": flow.get("course_interest"),
            "student_profile": flow.get("student_profile", {}),
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error getting flow status: {e}")
        return jsonify({"error": "Could not retrieve flow status"}), 500
    
    
@app.route("/set_student_info", methods=["POST"])
def set_student_info():
    """Set student information manually"""
    data = request.get_json()
    session_id = session.get('session_id', str(uuid.uuid4()))
    
    if 'session_id' not in session:
        session['session_id'] = session_id
    
    try:
        chatbot = get_or_create_chatbot(session_id)
        profile = chatbot.conversation_flow['student_profile']
        
        # Update student profile with provided information
        if 'name' in data:
            profile['name'] = data['name']
        if 'education_level' in data:
            profile['education_level'] = data['education_level']
        if 'preferred_courses' in data:
            profile['preferred_courses'] = data['preferred_courses']
        
        return jsonify({
            "message": "Student information updated",
            "student_profile": profile,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error setting student info: {e}")
        return jsonify({"error": "Could not update student information"}), 500

@app.route("/get_recommendations", methods=["GET"])
def get_recommendations():
    """Get contextual recommendations (stage-free)"""
    session_id = session.get('session_id')
    default_recommendations = [
        "Hello! How can I help you today?",
        "Tell me about available courses",
        "I want to know about admissions",
        "Show me campus facilities"
    ]
    if not session_id or session_id not in chatbot_sessions:
        return jsonify({
            "recommendations": default_recommendations,
            "session_id": session_id
        })
    try:
        chatbot = chatbot_sessions[session_id]
        # Use intent-based recommendations
        current_intent = chatbot.conversation_flow.get('current_intent', 'unknown')
        profile = chatbot.conversation_flow['student_profile']
        
        # Generate dynamic recommendations (if method exists)
        if hasattr(chatbot, 'get_contextual_recommendations'):
            recommendations = chatbot.get_contextual_recommendations(current_intent, profile)
        else:
            recommendations = default_recommendations

        return jsonify({
            "recommendations": recommendations[:6],
            "current_intent": current_intent,
            "course_interest": chatbot.conversation_flow.get('course_interest'),
            "session_id": session_id
        })
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({
            "recommendations": default_recommendations,
            "error": str(e),
            "session_id": session_id
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        active_sessions = len(chatbot_sessions)
        return jsonify({
            "status": "healthy",
            "active_sessions": active_sessions,
            "timestamp": datetime.now().isoformat(),
            "message": "Panjab University Chatbot is running successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/cleanup_sessions", methods=["POST"])
def cleanup_sessions():
    """Clean up inactive sessions (older than 1 hour)"""
    try:
        global chatbot_sessions
        
        sessions_to_keep = {}
        current_time = datetime.now()
        
        for session_id, chatbot in chatbot_sessions.items():
            session_start = chatbot.conversation_flow['session_start']
            session_age = (current_time - session_start).seconds
            
            # Keep sessions younger than 1 hour (3600 seconds)
            if session_age < 3600:
                sessions_to_keep[session_id] = chatbot
        
        cleaned_count = len(chatbot_sessions) - len(sessions_to_keep)
        chatbot_sessions = sessions_to_keep
        
        return jsonify({
            "message": f"Cleaned up {cleaned_count} inactive sessions",
            "active_sessions": len(chatbot_sessions),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Session cleanup failed: {str(e)}"}), 500

# Simple endpoint for backward compatibility
@app.route("/get_simple_response", methods=["POST"])
def get_simple_response():
    """Simple endpoint for basic chatbot functionality"""
    data = request.get_json()
    user_message = data.get("message", "")
    
    try:
        bot_reply = get_bot_response(user_message)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        print(f"Simple response error: {e}")
        return jsonify({"reply": "Sorry, I couldn't process your request right now."})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == "__main__":
    print("🎓 Starting Panjab University Chatbot Server...")
    print("📁 Make sure you have these files in the same directory:")
    print("   • flow_bot.py (chatbot logic)")
    print("   • intents.json or intents2.json (intent data)")
    print("   • templates/index1.html (web interface)")
    app.run(debug=True)


if __name__ == "__main__":
    print("🎓 Starting Panjab University Chatbot Server...")
    print("📁 Make sure you have these files in the same directory:")
    print("   • flow_bot.py (chatbot logic)")
    print("   • intents.json or intents2.json (intent data)")
    print("   • templates/index1.html (web interface)")
    port = int(os.environ.get("PORT", 5000))  # Render gives you a dynamic port
    print(f"🚀 Starting server on 0.0.0.0:{port}") 
    app.run(host="0.0.0.0", port=port)
