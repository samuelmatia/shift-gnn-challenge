"""
Generate leaderboard from evaluation results.
"""
import json
from datetime import datetime
from pathlib import Path

def format_datetime(iso_string):
    """Format ISO datetime string to human-readable format."""
    if not iso_string:
        return "Never"
    
    try:
        # Parse ISO format (handle both with and without timezone)
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        # Format: "December 20, 2025 at 22:56:05"
        return dt.strftime("%B %d, %Y at %H:%M:%S")
    except (ValueError, AttributeError):
        # If parsing fails, return as-is or try alternative format
        try:
            dt = datetime.strptime(iso_string.split('.')[0], '%Y-%m-%dT%H:%M:%S')
            return dt.strftime("%B %d, %Y at %H:%M:%S")
        except:
            return iso_string

def load_evaluation_results():
    """Load evaluation results."""
    results_file = Path(__file__).parent.parent / 'evaluation_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []

def load_existing_leaderboard():
    """Load existing leaderboard."""
    leaderboard_file = Path(__file__).parent.parent / 'leaderboard.json'
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            return json.load(f)
    return {"last_updated": None, "submissions": []}

def generate_leaderboard():
    """Generate leaderboard from all results."""
    results = load_evaluation_results()
    existing = load_existing_leaderboard()
    
    # Create a map of existing submissions
    existing_map = {sub['team']: sub for sub in existing.get('submissions', [])}
    
    # Process new results
    for result in results:
        team = result['team']
        scores = result['scores']
        
        entry = {
            'team': team,
            'submission_file': result['file'],
            'weighted_f1': scores.get('weighted_f1', 0.0),
            'overall_f1': scores.get('overall_macro_f1', 0.0),
            'rare_f1': scores.get('rare_transitions_f1', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update if better score or new team
        if team not in existing_map or entry['weighted_f1'] > existing_map[team]['weighted_f1']:
            existing_map[team] = entry
    
    # Convert to list and sort
    submissions = list(existing_map.values())
    submissions.sort(key=lambda x: x['weighted_f1'], reverse=True)
    
    leaderboard = {
        'last_updated': datetime.now().isoformat(),
        'submissions': submissions
    }
    
    # Save JSON
    leaderboard_file = Path(__file__).parent.parent / 'leaderboard.json'
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    # Generate HTML
    generate_html(leaderboard)
    
    print(f"Generated leaderboard with {len(submissions)} teams")
    return leaderboard

def generate_html(leaderboard):
    """Generate HTML leaderboard with animated graph background."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Challenge - Leaderboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            min-height: 100vh;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        
        #graph-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            opacity: 0.3;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            padding: 30px 0;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 0.8s ease-out;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        .header p {
            font-size: 1.3em;
            opacity: 0.9;
            color: #e0e0e0;
        }
        
        .last-updated {
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 30px;
            font-size: 0.95em;
            animation: fadeIn 1s ease-out 0.3s both;
        }
        
        .leaderboard {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 20px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        tbody tr {
            border-bottom: 1px solid #ecf0f1;
            transition: all 0.3s ease;
            animation: fadeInRow 0.5s ease-out both;
        }
        
        tbody tr:nth-child(even) {
            background: rgba(102, 126, 234, 0.02);
        }
        
        tbody tr:hover {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        
        tbody tr:last-child {
            border-bottom: none;
        }
        
        td {
            padding: 18px 20px;
        }
        
        .rank {
            font-weight: bold;
            font-size: 1.2em;
            width: 80px;
        }
        
        .rank-1 { 
            color: #f39c12;
        }
        .rank-2 { color: #95a5a6; }
        .rank-3 { color: #e67e22; }
        
        .medal {
            font-size: 1.5em;
            margin-right: 8px;
            display: inline-block;
        }
        
        .team-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1em;
        }
        
        .score {
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        
        .primary-score {
            color: #27ae60;
            font-size: 1.2em;
            font-weight: 700;
        }
        
        .empty {
            text-align: center;
            padding: 80px;
            color: #95a5a6;
            font-size: 1.2em;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1em;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            animation: fadeIn 1s ease-out 0.5s both;
        }
        
        .footer a {
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
            border-bottom: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .footer a:hover {
            color: white;
            border-bottom-color: rgba(255, 255, 255, 0.6);
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInRow {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes glow {
            from {
                filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5));
            }
            to {
                filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.8));
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
        }
        
        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-5px);
            }
        }
    </style>
</head>
<body>
    <canvas id="graph-canvas"></canvas>
    
    <div class="container">
        <div class="header">
            <h1>🏆 GNN Challenge Leaderboard</h1>
            <p>Role Transition Prediction in Temporal Networks</p>
        </div>
        <p class="last-updated">Last updated: """ + format_datetime(leaderboard.get("last_updated")) + """</p>
        
        <div class="leaderboard">
            <table>
                <thead>
                    <tr>
                        <th class="rank">Rank</th>
                        <th>Team</th>
                        <th class="score primary-score">Weighted Macro-F1</th>
                        <th class="score">Overall Macro-F1</th>
                        <th class="score">Rare Transitions F1</th>
                        <th>Submission Time</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    if not leaderboard.get("submissions"):
        html += """                    <tr>
                        <td colspan="6" class="empty">No submissions yet. Be the first! 🚀</td>
                    </tr>
"""
    else:
        for idx, entry in enumerate(leaderboard["submissions"], 1):
            rank_class = f"rank-{idx}" if idx <= 3 else ""
            medal = ""
            if idx == 1:
                medal = "🥇 "
            elif idx == 2:
                medal = "🥈 "
            elif idx == 3:
                medal = "🥉 "
            
            timestamp = entry.get("timestamp", "")
            if timestamp:
                timestamp = format_datetime(timestamp)
            
            html += f"""                    <tr style="animation-delay: {idx * 0.1}s;">
                        <td class="rank {rank_class}"><span class="medal">{medal}</span>{idx}</td>
                        <td class="team-name">{entry['team']}</td>
                        <td class="score primary-score">{entry['weighted_f1']:.6f}</td>
                        <td class="score">{entry['overall_f1']:.6f}</td>
                        <td class="score">{entry['rare_f1']:.6f}</td>
                        <td>{timestamp}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Submit your solution via Pull Request to appear on the leaderboard!</p>
            <p style="margin-top: 10px; font-size: 0.95em;">
                <a href="https://github.com/samuelmatia/gnn-role-transition-challenge" 
                   target="_blank" 
                   rel="noopener noreferrer">
                    🔗 View Repository on GitHub
                </a>
            </p>
        </div>
    </div>
    
    <script>
        // Animated Graph Background
        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Enhanced Graph Network Background - Fixed size with subtle animation
        const nodes = [];
        const nodeCount = 80;
        const connectionDistance = 280;
        const minDistance = 120;
        const maxDistance = 350;
        const springForce = 0.0001; // Force de rappel vers position initiale
        const repulsionForce = 0.01; // Répulsion minimale pour éviter les collisions
        const damping = 0.98; // Damping élevé pour mouvements subtils
        const animationAmplitude = 3; // Amplitude des petits mouvements
        
        // Initialize nodes with better distribution
        class Node {
            constructor() {
                this.initialX = Math.random() * canvas.width;
                this.initialY = Math.random() * canvas.height;
                this.x = this.initialX;
                this.y = this.initialY;
                this.vx = 0;
                this.vy = 0;
                this.radius = Math.random() * 2.5 + 1.5;
                this.baseHue = Math.random() * 60 + 240; // Purple/blue range
                this.hue = this.baseHue;
                this.brightness = 50 + Math.random() * 20;
                this.pulsePhase = Math.random() * Math.PI * 2;
                this.pulseSpeed = 0.01 + Math.random() * 0.01;
                this.oscillationPhase = Math.random() * Math.PI * 2;
                this.oscillationSpeed = 0.005 + Math.random() * 0.005;
                this.oscillationRadius = 2 + Math.random() * 3; // Rayon d'oscillation
            }
            
            update(others) {
                // Force de rappel vers la position initiale (spring)
                const dx = this.initialX - this.x;
                const dy = this.initialY - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 0.1) {
                    const spring = springForce * distance;
                    this.vx += (dx / distance) * spring;
                    this.vy += (dy / distance) * spring;
                }
                
                // Petite répulsion pour éviter les collisions
                for (let other of others) {
                    if (other === this) continue;
                    
                    const dx = other.x - this.x;
                    const dy = other.y - this.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < minDistance && dist > 0.1) {
                        const force = repulsionForce / (dist * dist + 1);
                        this.vx -= (dx / dist) * force;
                        this.vy -= (dy / dist) * force;
                    }
                }
                
                // Petits mouvements oscillatoires subtils
                this.oscillationPhase += this.oscillationSpeed;
                const oscX = Math.cos(this.oscillationPhase) * this.oscillationRadius;
                const oscY = Math.sin(this.oscillationPhase * 1.3) * this.oscillationRadius;
                
                // Appliquer damping et mettre à jour la position
                this.vx *= damping;
                this.vy *= damping;
                
                // Position avec oscillation subtile
                this.x = this.initialX + oscX + this.vx;
                this.y = this.initialY + oscY + this.vy;
                
                // Pulse animation pour la couleur
                this.pulsePhase += this.pulseSpeed;
                this.hue = this.baseHue + Math.sin(this.pulsePhase * 0.5) * 10;
            }
            
            draw(ctx) {
                // Outer glow with pulse
                const glowRadius = this.radius * 4;
                const gradient = ctx.createRadialGradient(
                    this.x, this.y, 0,
                    this.x, this.y, glowRadius
                );
                const pulse = Math.sin(this.pulsePhase) * 0.2 + 0.3;
                gradient.addColorStop(0, `hsla(${this.hue}, 80%, ${this.brightness}%, ${pulse * 0.6})`);
                gradient.addColorStop(0.5, `hsla(${this.hue}, 70%, ${this.brightness}%, ${pulse * 0.3})`);
                gradient.addColorStop(1, `hsla(${this.hue}, 60%, ${this.brightness}%, 0)`);
                
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(this.x, this.y, glowRadius, 0, Math.PI * 2);
                ctx.fill();
                
                // Inner core
                const coreGradient = ctx.createRadialGradient(
                    this.x, this.y, 0,
                    this.x, this.y, this.radius * 1.5
                );
                coreGradient.addColorStop(0, `hsl(${this.hue}, 90%, ${this.brightness + 10}%)`);
                coreGradient.addColorStop(1, `hsl(${this.hue}, 80%, ${this.brightness}%)`);
                
                ctx.fillStyle = coreGradient;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fill();
                
                // Highlight
                ctx.fillStyle = `hsla(${this.hue}, 100%, 90%, 0.6)`;
                ctx.beginPath();
                ctx.arc(this.x - this.radius * 0.3, this.y - this.radius * 0.3, this.radius * 0.4, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        // Create nodes
        for (let i = 0; i < nodeCount; i++) {
            nodes.push(new Node());
        }
        
        // Draw connections with improved visuals
        function drawConnections(ctx) {
            const connections = [];
            
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < connectionDistance) {
                        connections.push({
                            from: nodes[i],
                            to: nodes[j],
                            distance: distance,
                            strength: 1 - (distance / connectionDistance)
                        });
                    }
                }
            }
            
            // Sort by distance to draw closer connections on top
            connections.sort((a, b) => b.distance - a.distance);
            
            // Draw connections
            connections.forEach(conn => {
                const opacity = conn.strength * 0.4;
                const lineWidth = 0.5 + conn.strength * 1.5;
                
                // Create gradient for connection
                const gradient = ctx.createLinearGradient(
                    conn.from.x, conn.from.y,
                    conn.to.x, conn.to.y
                );
                gradient.addColorStop(0, `hsla(${conn.from.hue}, 70%, 60%, ${opacity})`);
                gradient.addColorStop(0.5, `hsla(${(conn.from.hue + conn.to.hue) / 2}, 80%, 65%, ${opacity * 1.2})`);
                gradient.addColorStop(1, `hsla(${conn.to.hue}, 70%, 60%, ${opacity})`);
                
                ctx.strokeStyle = gradient;
                ctx.lineWidth = lineWidth;
                ctx.shadowBlur = 3;
                ctx.shadowColor = `hsla(${(conn.from.hue + conn.to.hue) / 2}, 70%, 60%, ${opacity})`;
                ctx.beginPath();
                ctx.moveTo(conn.from.x, conn.from.y);
                ctx.lineTo(conn.to.x, conn.to.y);
                ctx.stroke();
                ctx.shadowBlur = 0;
            });
        }
        
        // Mouse interaction
        let mouseX = canvas.width / 2;
        let mouseY = canvas.height / 2;
        let targetMouseX = mouseX;
        let targetMouseY = mouseY;
        
        canvas.addEventListener('mousemove', (e) => {
            targetMouseX = e.clientX;
            targetMouseY = e.clientY;
        });
        
        // Smooth mouse tracking
        function updateMouse() {
            mouseX += (targetMouseX - mouseX) * 0.1;
            mouseY += (targetMouseY - mouseY) * 0.1;
        }
        
        // Animation loop
        function animate() {
            updateMouse();
            
            // Semi-transparent background for trail effect
            ctx.fillStyle = 'rgba(10, 14, 39, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Update nodes with physics
            nodes.forEach(node => {
                // Mouse repulsion (très subtile pour ne pas déplacer trop)
                const dx = node.x - mouseX;
                const dy = node.y - mouseY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 150 && distance > 0) {
                    const force = (150 - distance) / 150 * 0.01; // Réduite
                    node.vx += (dx / distance) * force;
                    node.vy += (dy / distance) * force;
                }
                
                node.update(nodes);
            });
            
            // Draw connections first (behind nodes)
            drawConnections(ctx);
            
            // Draw nodes on top
            nodes.forEach(node => {
                node.draw(ctx);
            });
            
            requestAnimationFrame(animate);
        }
        
        // Start animation
        animate();
    </script>
</body>
</html>"""
    
    html_file = Path(__file__).parent.parent / 'leaderboard.html'
    with open(html_file, 'w') as f:
        f.write(html)
    
    print(f"Generated {html_file}")

if __name__ == '__main__':
    generate_leaderboard()

