import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="🤖 RL Diet Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
def calculate_bmi(weight, height_cm):
    """Calculate BMI"""
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    """Get BMI category and icon"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal Weight", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    elif bmi < 35:
        return "Obese Level I", "🟠"
    elif bmi < 40:
        return "Obese Level II", "🔴"
    else:
        return "Obese Level III", "🔴"

def calculate_health_score(veg, water, exercise, screen, meals):
    """Calculate health score (0-100)"""
    score = 0
    
    # Vegetable intake (0-20 points)
    if veg >= 5:
        score += 20
    elif veg >= 3:
        score += 15
    elif veg >= 2:
        score += 10
    elif veg >= 1:
        score += 5
    
    # Water intake (0-20 points)
    if water >= 3:
        score += 20
    elif water >= 2:
        score += 15
    elif water >= 1:
        score += 10
    
    # Exercise (0-20 points)
    if exercise >= 5:
        score += 20
    elif exercise >= 3:
        score += 15
    elif exercise >= 1:
        score += 10
    
    # Screen time (0-20 points) - lower is better
    if screen <= 2:
        score += 20
    elif screen <= 4:
        score += 15
    elif screen <= 6:
        score += 10
    elif screen <= 8:
        score += 5
    
    # Meals per day (0-20 points)
    if meals >= 3 and meals <= 4:
        score += 20
    elif meals >= 2 and meals <= 5:
        score += 15
    elif meals >= 1 and meals <= 6:
        score += 10
    
    return min(score, 100)

def create_user_state(height, weight, age, gender, veg, water, exercise, screen, meals):
    """Create normalized state vector for RL agent"""
    bmi = calculate_bmi(weight, height)
    
    # Normalize to [0, 1] range
    bmi_norm = np.clip((bmi - 15) / (50 - 15), 0, 1)
    age_norm = np.clip((age - 16) / (80 - 16), 0, 1)
    gender_norm = 1 if gender == "Female" else 0
    exercise_norm = exercise / 7
    water_norm = (water - 1) / (5 - 1)
    veg_norm = veg / 5
    screen_norm = screen / 12
    meals_norm = (meals - 1) / (6 - 1)
    
    return np.array([bmi_norm, age_norm, gender_norm, exercise_norm, 
                    water_norm, veg_norm, screen_norm, meals_norm])

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class PolicyGradientAgent:
    """Simplified Policy Gradient Agent for Diet Recommendations"""
    
    def __init__(self, state_dim=8, action_dim=6, learning_rate=0.01, discount_factor=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Policy network parameters (simple linear model)
        self.theta = np.random.normal(0, 0.1, (state_dim, action_dim))
        
        # Training metrics
        self.training_rewards = []
        self.episode_rewards = []
    
    def get_action(self, state, training=True):
        """Select action using policy"""
        # Compute action probabilities using softmax
        logits = np.dot(state, self.theta)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        action_probs = exp_logits / np.sum(exp_logits)
        
        if training:
            # Sample action from policy
            action = np.random.choice(self.action_dim, p=action_probs)
            return action, action_probs
        else:
            # Greedy action selection
            action = np.argmax(action_probs)
            return action, action_probs
    
    def update(self, states, actions, rewards):
        """Update policy using REINFORCE algorithm"""
        # Compute discounted returns
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.discount_factor * discounted_return
            returns.insert(0, discounted_return)
        
        # Normalize returns
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Compute policy gradient
        gradient = np.zeros_like(self.theta)
        
        for i, (state, action, return_val) in enumerate(zip(states, actions, returns)):
            # Compute action probabilities
            logits = np.dot(state, self.theta)
            exp_logits = np.exp(logits - np.max(logits))
            action_probs = exp_logits / np.sum(exp_logits)
            
            # Compute gradient
            for a in range(self.action_dim):
                if a == action:
                    gradient[:, a] += return_val * (1 - action_probs[a]) * state
                else:
                    gradient[:, a] += return_val * (-action_probs[a]) * state
        
        # Update policy parameters
        self.theta += self.learning_rate * gradient
    
    def train(self, env, episodes=1000, verbose=True):
        """Train the Policy Gradient agent"""
        self.training_rewards = []
        self.episode_rewards = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for episode in range(episodes):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            
            for step in range(env.max_steps):
                action, _ = self.get_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update policy
            self.update(states, actions, rewards)
            
            self.episode_rewards.append(episode_reward)
            
            # Calculate average reward for last 100 episodes
            if len(self.episode_rewards) >= 100:
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.training_rewards.append(avg_reward)
            else:
                self.training_rewards.append(episode_reward)
            
            # Update progress
            if episode % 50 == 0:
                progress = episode / episodes
                progress_bar.progress(progress)
                status_text.text(f'Episode {episode}/{episodes} - Avg Reward: {self.training_rewards[-1]:.3f}')
        
        progress_bar.progress(1.0)
        status_text.text('Training completed!')
        
        return self.training_rewards

class DietRecommendationEnvironment:
    """Simplified Environment for Diet Recommendations"""
    
    def __init__(self, data=None):
        # State space: [BMI, age, gender, exercise, water, vegetables, screen_time, meals]
        self.state_dim = 8
        self.action_dim = 6
        
        # Action space: 6 different diet strategies
        self.actions = {
            0: "High Protein, Low Carb",
            1: "Balanced Mediterranean", 
            2: "Low Calorie, High Volume",
            3: "Intermittent Fasting",
            4: "Plant-Based Focus",
            5: "Keto-Inspired"
        }
        
        # State bounds for normalization
        self.state_bounds = {
            'bmi': (15, 50),
            'age': (16, 80),
            'gender': (0, 1),
            'exercise': (0, 7),
            'water': (0, 5),
            'vegetables': (0, 5),
            'screen_time': (0, 12),
            'meals': (1, 6)
        }
        
        # Load or generate data
        if data is not None:
            self.data = data
        else:
            self.data = self._load_or_generate_data()
            
        self.current_state = None
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = 50  # Reduced for faster training
        
    def _load_or_generate_data(self):
        """Load processed obesity data"""
        try:
            # Load the processed obesity data
            df = pd.read_csv('processed_obesity_data.csv')
            print(f"✅ Loaded processed obesity data: {len(df)} samples")
            print(f"📊 Available columns: {list(df.columns)}")
            
            # Check for required columns
            required_columns = ['BMI', 'Age', 'Gender', 'ActivityLevel', 'H2O', 'eatvege', 'TechFreq', 'MainMeal']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.stop()
            
            # Map columns to our format
            # Map ActivityLevel to exercise days per week
            activity_mapping = {
                'StayHome': 0,
                'Moderately Active': 3,
                'Active': 6
            }
            df['exercise_days'] = df['ActivityLevel'].map(activity_mapping)
            
            # Map TechFreq to screen time hours
            screen_mapping = {0: 0, 1: 6, 2: 12}
            df['screen_hours'] = df['TechFreq'].map(screen_mapping)
            
            processed_data = pd.DataFrame({
                'bmi': df['BMI'],
                'age': df['Age'],
                'gender': (df['Gender'] == 'Female').astype(int),
                'exercise': df['exercise_days'],
                'water': df['H2O'],
                'vegetables': df['eatvege'],
                'screen_time': df['screen_hours'],
                'meals': df['MainMeal']
            })
            
            print(f"✅ Data processed successfully. Shape: {processed_data.shape}")
            return processed_data
            
        except FileNotFoundError:
            st.error("❌ processed_obesity_data.csv not found! Please ensure the file is in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()
    
    def reset(self):
        """Reset environment to initial state"""
        user_data = self.data.sample(1).iloc[0]
        self.current_state = self._normalize_state(user_data)
        self.episode_reward = 0
        self.step_count = 0
        return self.current_state
    
    def _normalize_state(self, user_data):
        """Normalize user data to [0, 1] range"""
        try:
            normalized = np.zeros(self.state_dim)
            
            # BMI normalization
            normalized[0] = (user_data['bmi'] - self.state_bounds['bmi'][0]) / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])
            
            # Age normalization
            normalized[1] = (user_data['age'] - self.state_bounds['age'][0]) / (self.state_bounds['age'][1] - self.state_bounds['age'][0])
            
            # Gender
            normalized[2] = user_data['gender']
            
            # Exercise normalization
            normalized[3] = user_data['exercise'] / self.state_bounds['exercise'][1]
            
            # Water normalization
            normalized[4] = user_data['water'] / self.state_bounds['water'][1]
            
            # Vegetables normalization
            normalized[5] = user_data['vegetables'] / self.state_bounds['vegetables'][1]
            
            # Screen time normalization
            normalized[6] = user_data['screen_time'] / self.state_bounds['screen_time'][1]
            
            # Meals normalization
            normalized[7] = (user_data['meals'] - self.state_bounds['meals'][0]) / (self.state_bounds['meals'][1] - self.state_bounds['meals'][0])
            
            return np.clip(normalized, 0, 1)
            
        except KeyError as e:
            print(f"❌ Missing key in user_data: {e}")
            print(f"Available keys: {list(user_data.keys())}")
            raise
        except Exception as e:
            print(f"❌ Error normalizing state: {e}")
            print(f"User data: {user_data}")
            raise
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        if self.current_state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        reward = self._calculate_reward(action)
        next_state = self._simulate_state_transition(action)
        
        self.episode_reward += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        self.current_state = next_state
        return next_state, reward, done
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and current state"""
        # Get current BMI
        bmi = self.current_state[0] * (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0]) + self.state_bounds['bmi'][0]
        exercise = self.current_state[3] * (self.state_bounds['exercise'][1] - self.state_bounds['exercise'][0])
        vegetables = self.current_state[5] * (self.state_bounds['vegetables'][1] - self.state_bounds['vegetables'][0])
        
        # BMI-based reward (higher reward for closer to normal BMI range 18.5-24.9)
        if 18.5 <= bmi <= 24.9:
            # Perfect BMI range - highest reward
            bmi_reward = 1.0
        elif 17 <= bmi < 18.5 or 24.9 < bmi <= 26:
            # Close to normal range - high reward
            bmi_reward = 0.8
        elif 15 <= bmi < 17 or 26 < bmi <= 30:
            # Moderate distance from normal - medium reward
            bmi_reward = 0.6
        elif 30 < bmi <= 35:
            # Overweight/Obese I - lower reward
            bmi_reward = 0.4
        else:
            # Very far from normal - lowest reward
            bmi_reward = 0.2
        
        # Action effectiveness based on BMI compatibility
        action_effectiveness = {
            0: 0.8,  # High Protein, Low Carb
            1: 0.9,  # Balanced Mediterranean
            2: 0.7,  # Low Calorie, High Volume
            3: 0.6,  # Intermittent Fasting
            4: 0.8,  # Plant-Based Focus
            5: 0.5   # Keto-Inspired
        }
        
        base_reward = action_effectiveness[action]
        
        # BMI-specific action adjustments
        if bmi > 30 and action in [0, 2]:  # Good for obese
            base_reward += 0.2
        elif bmi < 20 and action in [1, 4]:  # Good for underweight
            base_reward += 0.2
        elif bmi > 30 and action in [5, 3]:  # Too extreme for obese
            base_reward -= 0.3
        elif bmi < 18.5 and action in [2, 3, 5]:  # Too restrictive for underweight
            base_reward -= 0.4
            
        # Lifestyle factor adjustments
        if exercise > 4 and action in [0, 1]:  # Good for active people
            base_reward += 0.1
        elif exercise < 2 and action in [5, 3]:  # Too extreme for sedentary
            base_reward -= 0.2
            
        if vegetables > 3 and action == 4:  # Plant-based for veggie lovers
            base_reward += 0.1
        elif vegetables < 1 and action in [5, 3]:  # Too restrictive for low veggie intake
            base_reward -= 0.2
        
        # Combine BMI reward with action effectiveness
        total_reward = (bmi_reward * 0.6) + (base_reward * 0.4)
        
        # Add small noise for variability
        noise = np.random.normal(0, 0.05)
        return total_reward + noise
    
    def _simulate_state_transition(self, action):
        """Simulate state changes based on action"""
        next_state = self.current_state.copy()
        
        # Get current BMI for intelligent transitions
        current_bmi = next_state[0] * (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0]) + self.state_bounds['bmi'][0]
        
        # Calculate BMI change based on current BMI and action (increased for longer timeline)
        if action == 0:  # High Protein, Low Carb
            if current_bmi > 25:  # If overweight/obese, reduce BMI
                bmi_change = -0.08  # Increased from -0.02
            elif current_bmi < 18.5:  # If underweight, slight increase
                bmi_change = 0.04  # Increased from 0.01
            else:  # Normal weight, maintain
                bmi_change = -0.02  # Increased from -0.005
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[3] = min(1, next_state[3] + 0.08)  # Exercise increase (increased from 0.02)
            
        elif action == 1:  # Balanced Mediterranean
            if current_bmi > 25:  # If overweight/obese, reduce BMI
                bmi_change = -0.06  # Increased from -0.015
            elif current_bmi < 18.5:  # If underweight, slight increase
                bmi_change = 0.04  # Increased from 0.01
            else:  # Normal weight, maintain
                bmi_change = -0.02  # Increased from -0.005
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[5] = min(1, next_state[5] + 0.04)  # Vegetable increase (increased from 0.01)
            next_state[4] = min(1, next_state[4] + 0.04)  # Water increase (increased from 0.01)
            
        elif action == 2:  # Low Calorie, High Volume
            if current_bmi > 25:  # If overweight/obese, reduce BMI
                bmi_change = -0.10  # Increased from -0.025
            elif current_bmi < 18.5:  # If underweight, avoid further reduction
                bmi_change = 0.02  # Increased from 0.005
            else:  # Normal weight, slight reduction
                bmi_change = -0.04  # Increased from -0.01
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[5] = min(1, next_state[5] + 0.08)  # Vegetable increase (increased from 0.02)
            
        elif action == 3:  # Intermittent Fasting
            if current_bmi > 25:  # If overweight/obese, reduce BMI
                bmi_change = -0.08  # Increased from -0.02
            elif current_bmi < 18.5:  # If underweight, avoid fasting
                bmi_change = 0.04  # Increased from 0.01
            else:  # Normal weight, slight reduction
                bmi_change = -0.04  # Increased from -0.01
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[7] = max(0, next_state[7] - 0.04)  # Meal reduction (increased from 0.01)
            
        elif action == 4:  # Plant-Based Focus
            if current_bmi > 25:  # If overweight/obese, reduce BMI
                bmi_change = -0.04  # Increased from -0.01
            elif current_bmi < 18.5:  # If underweight, slight increase
                bmi_change = 0.04  # Increased from 0.01
            else:  # Normal weight, maintain
                bmi_change = -0.02  # Increased from -0.005
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[5] = min(1, next_state[5] + 0.12)  # Vegetable increase (increased from 0.03)
            
        elif action == 5:  # Keto-Inspired
            if current_bmi > 30:  # If obese, reduce BMI
                bmi_change = -0.12  # Increased from -0.03
            elif current_bmi > 25:  # If overweight, reduce BMI
                bmi_change = -0.08  # Increased from -0.02
            elif current_bmi < 18.5:  # If underweight, avoid keto
                bmi_change = 0.04  # Increased from 0.01
            else:  # Normal weight, slight reduction
                bmi_change = -0.04  # Increased from -0.01
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.state_bounds['bmi'][1] - self.state_bounds['bmi'][0])))
            next_state[7] = max(0, next_state[7] - 0.08)  # Meal reduction (increased from 0.02)
        
        return next_state

def get_diet_recommendations():
    """Get detailed diet recommendations"""
    return {
        0: {  # High Protein, Low Carb
            "name": "High Protein, Low Carb",
            "description": "Focus on lean proteins with minimal carbohydrates",
            "foods": [
                "🥩 Lean meats: chicken breast, turkey, lean beef",
                "🐟 Fatty fish: salmon, mackerel, sardines",
                "🥚 Eggs and egg whites",
                "🧀 Greek yogurt, cottage cheese",
                "🥜 Nuts and seeds in moderation",
                "🥬 Non-starchy vegetables: spinach, broccoli, kale",
                "🥑 Avocados for healthy fats"
            ],
            "avoid": [
                "🍞 Bread, pasta, rice",
                "🍰 Sugary foods and desserts",
                "🍟 Processed snacks",
                "🍺 Alcoholic beverages"
            ],
            "tips": [
                "Aim for 1.2-1.6g protein per kg body weight",
                "Keep carbs under 50g per day",
                "Stay hydrated with water and herbal teas",
                "Consider intermittent fasting windows"
            ]
        },
        1: {  # Balanced Mediterranean
            "name": "Balanced Mediterranean",
            "description": "Balanced macronutrients with Mediterranean principles",
            "foods": [
                "🐟 Fish and seafood 2-3 times per week",
                "🥜 Nuts, seeds, and olive oil daily",
                "🍇 Fresh fruits and vegetables",
                "🌾 Whole grains: quinoa, brown rice, oats",
                "🥛 Greek yogurt and cheese in moderation",
                "🍷 Red wine in moderation (optional)",
                "🌿 Herbs and spices for flavor"
            ],
            "avoid": [
                "🍔 Processed meats",
                "🍰 Refined sugars and desserts",
                "🍟 Fried foods",
                "🥤 Sugary beverages"
            ],
            "tips": [
                "Use olive oil as primary fat source",
                "Include fish 2-3 times per week",
                "Focus on plant-based foods",
                "Enjoy meals with family and friends"
            ]
        },
        2: {  # Low Calorie, High Volume
            "name": "Low Calorie, High Volume",
            "description": "High-volume, low-calorie foods for satiety",
            "foods": [
                "🥬 Leafy greens: spinach, lettuce, kale",
                "🥒 Non-starchy vegetables: cucumbers, celery, bell peppers",
                "🍓 Berries and low-sugar fruits",
                "🥣 Broth-based soups",
                "🥗 Large salads with lean protein",
                "💧 Water-rich foods: watermelon, cucumber",
                "🥕 Crudités with hummus"
            ],
            "avoid": [
                "🍕 High-calorie dense foods",
                "🍰 Desserts and sweets",
                "🥤 Caloric beverages",
                "🍟 Fried and processed foods"
            ],
            "tips": [
                "Fill half your plate with vegetables",
                "Eat slowly and mindfully",
                "Drink water before meals",
                "Use smaller plates to control portions"
            ]
        },
        3: {  # Intermittent Fasting
            "name": "Intermittent Fasting",
            "description": "Time-restricted eating patterns",
            "foods": [
                "🥩 High-quality proteins during eating window",
                "🥬 Nutrient-dense vegetables",
                "🥑 Healthy fats: avocado, nuts, olive oil",
                "🍓 Low-sugar fruits",
                "🥚 Eggs and lean proteins",
                "🌾 Complex carbohydrates in moderation",
                "💧 Plenty of water and herbal teas"
            ],
            "avoid": [
                "🍰 Sugary foods during eating window",
                "🥤 Caloric beverages during fasting",
                "🍟 Processed and junk foods",
                "🍺 Alcohol (can break fast)"
            ],
            "tips": [
                "Start with 12:12 (12 hours eating, 12 hours fasting)",
                "Gradually increase to 16:8 or 18:6",
                "Stay hydrated during fasting periods",
                "Focus on nutrient density during eating windows"
            ]
        },
        4: {  # Plant-Based Focus
            "name": "Plant-Based Focus",
            "description": "Primarily plant-based nutrition",
            "foods": [
                "🥬 Leafy greens and vegetables",
                "🍎 Fresh fruits",
                "🌾 Whole grains: quinoa, brown rice, oats",
                "🥜 Nuts, seeds, and legumes",
                "🥑 Plant-based fats: avocado, olive oil",
                "🥛 Plant milks and yogurts",
                "🍄 Mushrooms and plant proteins"
            ],
            "avoid": [
                "🥩 Animal products",
                "🍳 Eggs and dairy",
                "🍔 Processed plant-based meats",
                "🍰 Vegan desserts high in sugar"
            ],
            "tips": [
                "Ensure adequate protein from plant sources",
                "Consider B12 supplementation",
                "Include variety of colorful vegetables",
                "Focus on whole, unprocessed foods"
            ]
        },
        5: {  # Keto-Inspired
            "name": "Keto-Inspired",
            "description": "Very low carbohydrate, high fat approach",
            "foods": [
                "🥩 Fatty cuts of meat",
                "🐟 Fatty fish and seafood",
                "🥑 Avocados and olive oil",
                "🥜 Nuts and seeds",
                "🧀 High-fat dairy products",
                "🥬 Non-starchy vegetables",
                "🥚 Eggs and egg yolks"
            ],
            "avoid": [
                "🍞 All grains and bread",
                "🍎 Most fruits (except berries)",
                "🍯 Sugars and sweeteners",
                "🍺 Alcoholic beverages"
            ],
            "tips": [
                "Keep carbs under 20g per day",
                "Focus on healthy fats",
                "Monitor ketone levels if desired",
                "Stay hydrated and supplement electrolytes"
            ]
        }
    }

class RecommendationImpactPredictor:
    """Predicts the impact of following vs not following RL recommendations"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
    def predict_following_recommendation(self, current_state, steps_ahead=90):
        """Predict outcomes when following the RL recommendation"""
        # Create a copy of the environment for prediction
        pred_env = DietRecommendationEnvironment()
        pred_env.current_state = current_state.copy()
        
        state = current_state.copy()
        predicted_states = [state.copy()]
        total_reward = 0
        
        for step in range(steps_ahead):
            # Get RL recommendation
            action, _ = self.agent.get_action(state, training=False)
            
            # Simulate following the recommendation
            next_state, reward, done = pred_env.step(action)
            state = next_state
            predicted_states.append(state.copy())
            total_reward += reward
            
            if done:
                break
                
        return predicted_states, total_reward
    
    def predict_not_following_recommendation(self, current_state, steps_ahead=90):
        """Predict outcomes when NOT following the RL recommendation"""
        # Create a copy of the environment for prediction
        pred_env = DietRecommendationEnvironment()
        pred_env.current_state = current_state.copy()
        
        state = current_state.copy()
        predicted_states = [state.copy()]
        total_reward = 0
        
        for step in range(steps_ahead):
            # Get RL recommendation
            recommended_action, _ = self.agent.get_action(state, training=False)
            
            # Simulate "not following recommendation" by making poor dietary choices
            action = self._get_poor_diet_choice(recommended_action, state)
            
            # Use a different state transition that simulates poor choices
            next_state = self._simulate_poor_state_transition(state, action)
            reward = self._calculate_poor_reward(action, state)
            
            state = next_state
            predicted_states.append(state.copy())
            total_reward += reward
            
            if step >= steps_ahead - 1:
                break
                
        return predicted_states, total_reward
    
    def _simulate_poor_state_transition(self, current_state, action):
        """Simulate state changes when making poor dietary choices"""
        next_state = current_state.copy()
        
        # Get current BMI for poor transitions
        current_bmi = next_state[0] * (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0]) + self.env.state_bounds['bmi'][0]
        
        # Poor BMI changes - realistic outcomes when not following recommendations
        if action == 0:  # High Protein, Low Carb (poor choice for some)
            if current_bmi < 18.5:  # If underweight, make it worse
                bmi_change = -0.12  # Further weight loss (bad for underweight)
            elif current_bmi > 25:  # If overweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal weight, slight increase
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            next_state[3] = max(0, next_state[3] - 0.04)  # Exercise decrease
            
        elif action == 1:  # Balanced Mediterranean (not following = poor choices)
            if current_bmi > 25:  # If overweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal/underweight, weight gain
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            next_state[5] = max(0, next_state[5] - 0.04)  # Vegetable decrease
            
        elif action == 2:  # Low Calorie, High Volume (not following = poor choices)
            if current_bmi < 18.5:  # If underweight, make it worse
                bmi_change = -0.08  # Further weight loss (bad for underweight)
            elif current_bmi > 25:  # If overweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal weight, weight gain
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            
        elif action == 3:  # Intermittent Fasting (not following = poor choices)
            if current_bmi < 18.5:  # If underweight, make it worse
                bmi_change = -0.08  # Further weight loss (bad for underweight)
            elif current_bmi > 25:  # If overweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal weight, weight gain
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            next_state[7] = min(1, next_state[7] + 0.04)  # Meal increase (more eating)
            
        elif action == 4:  # Plant-Based Focus (not following = poor choices)
            if current_bmi < 18.5:  # If underweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            elif current_bmi > 25:  # If overweight, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal weight, weight gain
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            
        elif action == 5:  # Keto-Inspired (not following = poor choices)
            if current_bmi < 18.5:  # If underweight, make it worse
                bmi_change = -0.12  # Further weight loss (bad for underweight)
            elif current_bmi > 30:  # If obese, no improvement or slight gain
                bmi_change = 0.02  # Slight weight gain (no improvement)
            else:  # Normal/overweight, weight gain
                bmi_change = 0.04  # Weight gain
            next_state[0] = max(0, min(1, next_state[0] + bmi_change / (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0])))
            next_state[7] = min(1, next_state[7] + 0.04)  # Meal increase (more eating)
        
        return next_state
    
    def _calculate_poor_reward(self, action, state):
        """Calculate lower reward for poor dietary choices"""
        # Get current BMI
        bmi = state[0] * (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0]) + self.env.state_bounds['bmi'][0]
        
        # Lower base reward for poor choices
        poor_action_effectiveness = {
            0: 0.3,  # High Protein, Low Carb (poor choice)
            1: 0.4,  # Balanced Mediterranean (less effective)
            2: 0.2,  # Low Calorie, High Volume (poor choice)
            3: 0.2,  # Intermittent Fasting (poor choice)
            4: 0.3,  # Plant-Based Focus (less effective)
            5: 0.1   # Keto-Inspired (poor choice)
        }
        
        base_reward = poor_action_effectiveness[action]
        
        # BMI-based reward (same as good choices but with lower multiplier)
        if 18.5 <= bmi <= 24.9:
            bmi_reward = 1.0
        elif 17 <= bmi < 18.5 or 24.9 < bmi <= 26:
            bmi_reward = 0.8
        elif 15 <= bmi < 17 or 26 < bmi <= 30:
            bmi_reward = 0.6
        elif 30 < bmi <= 35:
            bmi_reward = 0.4
        else:
            bmi_reward = 0.2
        
        # Combine with lower weights for poor choices
        total_reward = (bmi_reward * 0.3) + (base_reward * 0.7)
        
        # Add small noise
        noise = np.random.normal(0, 0.05)
        return total_reward + noise
    
    def _get_poor_diet_choice(self, recommended_action, current_state):
        """Get a poor dietary choice that represents not following recommendations"""
        # Get current BMI to determine appropriate "bad" choice
        bmi = current_state[0] * (self.env.state_bounds['bmi'][1] - self.env.state_bounds['bmi'][0]) + self.env.state_bounds['bmi'][0]
        
        # Define poor choices based on current health status
        if bmi < 18.5:  # Underweight - should gain weight healthily
            poor_choices = [2, 3, 5]  # Low Calorie, Intermittent Fasting, Keto
        elif bmi < 25:  # Normal weight - should maintain
            poor_choices = [5, 3]  # Keto, Intermittent Fasting
        elif bmi < 30:  # Overweight - should lose weight
            poor_choices = [1, 4]  # Mediterranean, Plant-Based (less effective for weight loss)
        else:  # Obese - should lose weight significantly
            poor_choices = [5, 3]  # Keto, Intermittent Fasting (too extreme for obese)
        
        # Choose a poor action that's different from recommendation
        available_poor_choices = [a for a in poor_choices if a != recommended_action]
        if available_poor_choices:
            return available_poor_choices[0]
        else:
            return poor_choices[0]

def main():
    """Main Streamlit app"""
    
    # Define helper functions locally to ensure they're in scope
    def get_bmi_category_local(bmi):
        """Get BMI category and icon - local function"""
        if bmi < 18.5:
            return "Underweight", "🔵"
        elif bmi < 25:
            return "Normal Weight", "🟢"
        elif bmi < 30:
            return "Overweight", "🟡"
        elif bmi < 35:
            return "Obese Level I", "🟠"
        elif bmi < 40:
            return "Obese Level II", "🔴"
        else:
            return "Obese Level III", "🔴"
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Diet Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Policy Gradient Reinforcement Learning")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Training section
        st.subheader("🤖 Train RL Agent")
        if st.button("🚀 Train Policy Gradient Agent", type="primary"):
            with st.spinner("Training RL agent... This may take a few minutes."):
                # Create environment
                env = DietRecommendationEnvironment()
                
                # Create agent
                agent = PolicyGradientAgent()
                
                # Train agent
                training_rewards = agent.train(env, episodes=500, verbose=False)
                
                # Store in session state
                st.session_state.agent = agent
                st.session_state.env = env
                st.session_state.trained = True
                
                st.success("✅ Agent trained successfully!")
                
                # Show training progress
                fig = px.line(x=range(len(training_rewards)), y=training_rewards,
                            title="Training Progress", labels={'x': 'Episodes', 'y': 'Average Reward'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Status
        if st.session_state.trained:
            st.success("✅ Agent is trained and ready!")
        else:
            st.warning("⚠️ Please train the agent first")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Your Health Profile")
        
        # Health inputs
        height = st.slider("Height (cm)", 120, 220, 170)
        weight = st.slider("Weight (kg)", 30, 150, 70)
        age = st.slider("Age", 16, 80, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("🏃 Lifestyle Assessment")
        veg = st.slider("Vegetable Intake (servings/day)", 0, 5, 2)
        water = st.slider("Water Intake (liters/day)", 1, 5, 2)
        exercise = st.slider("Exercise (days/week)", 0, 7, 2)
        screen = st.slider("Screen Time (hours/day)", 0, 12, 3)
        meals = st.slider("Main Meals (per day)", 1, 6, 3)
        
        # Calculate metrics
        if st.button("🧮 Calculate Health Metrics", type="primary"):
            try:
                bmi = calculate_bmi(weight, height)
                category_str, category_icon = get_bmi_category_local(bmi)
                health_score = calculate_health_score(veg, water, exercise, screen, meals)
                
                # Store in session state
                st.session_state.bmi = bmi
                st.session_state.category = category_str
                st.session_state.category_icon = category_icon
                st.session_state.health_score = health_score
                st.session_state.gender = gender
                st.session_state.user_state = create_user_state(height, weight, age, gender, veg, water, exercise, screen, meals)
                
                st.success("✅ Health metrics calculated successfully!")
            except Exception as e:
                st.error(f"❌ Error calculating health metrics: {str(e)}")
                st.info("Please check your input values and try again.")
    
    with col2:
        st.header("📈 Health Analysis")
        
        if 'bmi' in st.session_state:
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("BMI", f"{st.session_state.bmi:.1f}")
            with col_b:
                st.metric("Category", f"{st.session_state.category_icon} {st.session_state.category}")
            with col_c:
                st.metric("Health Score", f"{st.session_state.health_score}/100")
            
            # BMI visualization
            fig = go.Figure()
            
            # BMI categories
            categories = [
                ("Underweight", 0, 18.5, "lightblue"),
                ("Normal", 18.5, 24.9, "lightgreen"),
                ("Overweight", 25, 29.9, "yellow"),
                ("Obese I", 30, 34.9, "orange"),
                ("Obese II", 35, 39.9, "red"),
                ("Obese III", 40, 60, "darkred")
            ]
            
            for label, start, end, color in categories:
                fig.add_trace(go.Bar(
                    x=[end - start],
                    y=[label],
                    orientation='h',
                    marker_color=color,
                    name=label,
                    showlegend=False
                ))
            
            # Current BMI marker
            fig.add_vline(x=st.session_state.bmi, line_dash="dash", line_color="black", 
                         annotation_text=f"Your BMI: {st.session_state.bmi:.1f}")
            
            fig.update_layout(
                title="BMI Categories",
                xaxis_title="BMI Range",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Lifestyle chart
            lifestyle_data = {
                'Metric': ['Vegetables', 'Water', 'Exercise', 'Screen Time', 'Meals'],
                'Value': [veg, water, exercise, screen, meals],
                'Ideal': [4, 3, 5, 2, 3]  # Ideal values
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Your Value',
                x=lifestyle_data['Metric'],
                y=lifestyle_data['Value'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Ideal Value',
                x=lifestyle_data['Metric'],
                y=lifestyle_data['Ideal'],
                marker_color='lightgreen',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Lifestyle Habits vs Ideal",
                yaxis_title="Value",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # RL Recommendation Section
    if st.session_state.trained and 'user_state' in st.session_state:
        st.header("🤖 AI Diet Recommendation")
        
        # Button section
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🎯 Get Personalized Recommendation", type="primary"):
                # Get recommendation from RL agent
                action, action_probs = st.session_state.agent.get_action(st.session_state.user_state, training=False)
                
                # Get detailed recommendations
                recommendations = get_diet_recommendations()
                rec = recommendations[action]
                
                # Store in session state for prediction
                st.session_state.current_action = action
                st.session_state.current_action_probs = action_probs
                st.session_state.current_recommendation = rec
                st.session_state.show_recommendation = True
        
        with col_btn2:
            if st.button("🔮 Predict Recommendation Impact", type="secondary"):
                if 'current_action' not in st.session_state:
                    st.warning("Please get a recommendation first!")
                else:
                    st.session_state.show_prediction = True
        
        with col_btn3:
            if st.button("👤 Visualize Body Type", type="secondary"):
                if 'bmi' not in st.session_state:
                    st.warning("Please calculate BMI first!")
                else:
                    st.session_state.show_body_type = True
        
        # Display recommendation if available
        if 'show_recommendation' in st.session_state and st.session_state.show_recommendation:
            st.markdown("---")
            st.subheader("🎯 Your Personalized Recommendation")
            
            rec = st.session_state.current_recommendation
            action_probs = st.session_state.current_action_probs
            action = st.session_state.current_action
            
            # Display recommendation
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🎯 Recommended Diet: {rec['name']}</h3>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Confidence:</strong> {action_probs[action]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed recommendations in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Recommended Foods")
                for food in rec['foods']:
                    st.write(food)
                
                st.subheader("❌ Foods to Avoid")
                for food in rec['avoid']:
                    st.write(food)
            
            with col2:
                st.subheader("💡 Helpful Tips")
                for tip in rec['tips']:
                    st.write(f"• {tip}")
                
                # Action probabilities
                st.subheader("🧠 AI Confidence Levels")
                prob_data = pd.DataFrame({
                    'Diet Strategy': [get_diet_recommendations()[i]['name'] for i in range(6)],
                    'Confidence': action_probs
                }).sort_values('Confidence', ascending=True)
                
                fig = px.bar(prob_data, x='Confidence', y='Diet Strategy', 
                           orientation='h', title="AI Confidence in Each Strategy")
                st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction if requested
        if 'show_prediction' in st.session_state and st.session_state.show_prediction:
            st.markdown("---")
            st.subheader("🔮 What Happens If You Follow vs Don't Follow the Recommendation?")
            
            # Simple explanation
            st.info("""
            **This prediction shows you what could happen to your health over the next 90 days (3 months):**
            - 🟢 **Green line**: If you follow the AI recommendation
            - 🔴 **Red line**: If you ignore the recommendation and make poor food choices
            """)
            
            with st.spinner("🔮 Analyzing your health outcomes..."):
                # Create predictor
                predictor = RecommendationImpactPredictor(st.session_state.env, st.session_state.agent)
                
                # Get predictions
                try:
                    follow_states, follow_reward = predictor.predict_following_recommendation(st.session_state.user_state, steps_ahead=90)
                    not_follow_states, not_follow_reward = predictor.predict_not_following_recommendation(st.session_state.user_state, steps_ahead=90)
                    
                    # Extract BMI progression
                    follow_bmi = []
                    not_follow_bmi = []
                    
                    for state in follow_states:
                        bmi = state[0] * (st.session_state.env.state_bounds['bmi'][1] - st.session_state.env.state_bounds['bmi'][0]) + st.session_state.env.state_bounds['bmi'][0]
                        follow_bmi.append(bmi)
                        
                    for state in not_follow_states:
                        bmi = state[0] * (st.session_state.env.state_bounds['bmi'][1] - st.session_state.env.state_bounds['bmi'][0]) + st.session_state.env.state_bounds['bmi'][0]
                        not_follow_bmi.append(bmi)
                    
                    # Ensure we have data to plot
                    if len(follow_bmi) == 0 or len(not_follow_bmi) == 0:
                        st.error("❌ Prediction failed - no data generated")
                    else:
                        # Calculate key metrics
                        current_bmi = follow_bmi[0]
                        follow_final_bmi = follow_bmi[-1]
                        not_follow_final_bmi = not_follow_bmi[-1]
                        
                        bmi_change_follow = follow_final_bmi - current_bmi
                        bmi_change_not_follow = not_follow_final_bmi - current_bmi
                        
                        # Simple summary first
                        st.markdown("### 📊 **Quick Summary**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Your Current BMI", 
                                f"{current_bmi:.1f}",
                                help="This is your starting point"
                            )
                        
                        with col2:
                            follow_direction = "📉 Better" if bmi_change_follow < 0 else "📈 Worse" if bmi_change_follow > 0 else "➡️ Same"
                            st.metric(
                                "Following Recommendation", 
                                f"{follow_final_bmi:.1f}",
                                delta=f"{bmi_change_follow:+.1f}",
                                help=f"BMI after 30 days: {follow_direction}"
                            )
                        
                        with col3:
                            not_follow_direction = "📉 Better" if bmi_change_not_follow < 0 else "📈 Worse" if bmi_change_not_follow > 0 else "➡️ Same"
                            st.metric(
                                "Not Following", 
                                f"{not_follow_final_bmi:.1f}",
                                delta=f"{bmi_change_not_follow:+.1f}",
                                help=f"BMI after 30 days: {not_follow_direction}"
                            )
                        
                        # Simple BMI progression chart
                        st.markdown("### 📈 **Your BMI Over Time**")
                        
                        # Create simple BMI chart
                        fig_bmi = go.Figure()
                        
                        days = list(range(len(follow_bmi)))
                        fig_bmi.add_trace(go.Scatter(
                            x=days, y=follow_bmi, 
                            name='✅ Following Recommendation',
                            line=dict(color='#2ECC71', width=4),
                            mode='lines+markers',
                            marker=dict(size=6)
                        ))
                        fig_bmi.add_trace(go.Scatter(
                            x=days, y=not_follow_bmi, 
                            name='❌ Not Following',
                            line=dict(color='#E74C3C', width=4, dash='dash'),
                            mode='lines+markers',
                            marker=dict(size=6)
                        ))
                        
                        # Add healthy BMI range
                        fig_bmi.add_hline(y=18.5, line_dash="dot", line_color="blue", 
                                        annotation_text="Healthy BMI Range", annotation_position="bottom right")
                        fig_bmi.add_hline(y=24.9, line_dash="dot", line_color="blue")
                        
                        fig_bmi.update_layout(
                            title="Your BMI Prediction Over 90 Days (3 Months)",
                            xaxis_title="Days (0-90)",
                            yaxis_title="BMI",
                            height=400,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_bmi, use_container_width=True)
                        
                        # Simple comparison
                        st.markdown("### ⚖️ **The Difference**")
                        
                        bmi_difference = follow_final_bmi - not_follow_final_bmi
                        
                        if abs(bmi_difference) < 1.0:
                            st.warning("⚠️ **Small Difference**: Both paths lead to similar results over 3 months")
                        elif bmi_difference < 0:
                            st.success(f"🎉 **Following is Better**: You'll be {abs(bmi_difference):.1f} BMI points healthier after 3 months!")
                        else:
                            st.error(f"⚠️ **Not Following is Worse**: You'll be {bmi_difference:.1f} BMI points unhealthier after 3 months")
                        
                        # Simple explanation
                        st.markdown("### 💡 **What This Means**")
                        
                        if bmi_change_follow < 0 and bmi_change_not_follow < 0:
                            st.info("""
                            **Weight loss scenario over 3 months:**
                            - ✅ Following the recommendation helps you lose weight effectively
                            - ⚠️ Not following leads to further weight loss (bad for underweight users)
                            """)
                        elif bmi_change_follow > 0 and bmi_change_not_follow > 0:
                            st.warning("""
                            **Weight gain scenario over 3 months:**
                            - ⚠️ Following the recommendation leads to controlled, healthy weight gain
                            - ❌ Not following will cause more significant weight gain
                            """)
                        elif bmi_change_follow < 0 and bmi_change_not_follow > 0:
                            st.success("""
                            **Perfect scenario over 3 months:**
                            - ✅ Following the recommendation helps you lose weight and get healthier
                            - ❌ Not following will make you gain weight and become less healthy
                            - 🎯 The AI recommendation is clearly beneficial for your goals!
                            """)
                        else:
                            st.info("""
                            **The recommendation helps you over 3 months:**
                            - ✅ Maintain or improve your current health
                            - ⚠️ Not following could lead to health decline
                            """)
                        
                        # Health category explanation
                        current_cat, current_icon = get_bmi_category(current_bmi)
                        follow_cat, follow_icon = get_bmi_category(follow_final_bmi)
                        not_follow_cat, not_follow_icon = get_bmi_category(not_follow_final_bmi)
                        
                        st.markdown("### 🏥 **Health Category Changes**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Starting:** {current_icon} {current_cat}")
                        with col2:
                            st.markdown(f"**Following:** {follow_icon} {follow_cat}")
                        with col3:
                            st.markdown(f"**Not Following:** {not_follow_icon} {not_follow_cat}")
                        
                        if follow_cat != not_follow_cat:
                            st.success(f"🎯 **Important**: Following the recommendation could change your health category from {not_follow_cat} to {follow_cat}!")
                        
                        # Final recommendation
                        st.markdown("### 🎯 **Our Recommendation**")
                        
                        if bmi_difference < -0.5:
                            st.success("""
                            **Strongly recommend following the AI suggestion!** 
                            
                            The benefits are clear - you'll be significantly healthier by following the recommendation.
                            """)
                        elif bmi_difference < 0:
                            st.info("""
                            **We recommend following the AI suggestion.** 
                            
                            While the difference is modest, every step toward better health counts.
                            """)
                        else:
                            st.warning("""
                            **Consider the AI suggestion carefully.** 
                            
                            The recommendation may not be the best fit for your current situation.
                            """)
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please try getting a recommendation first, then try the prediction again.")
        
        # Display body type visualization if requested
        if 'show_body_type' in st.session_state and st.session_state.show_body_type:
            st.markdown("---")
            st.subheader("👤 Your Body Type Visualization")
            
            if 'bmi' in st.session_state and 'category' in st.session_state:
                # Create body type visualization
                bmi = st.session_state.bmi
                category = st.session_state.category
                gender = st.session_state.get('gender', 'Male')  # Default to Male if not set
                
                # Create the visualization
                fig, ax = plt.subplots(figsize=(8, 10))
                
                # Determine body color based on BMI category
                if category == "Underweight":
                    body_color = "#87CEEB"  # Light blue
                elif category == "Normal Weight":
                    body_color = "#90EE90"  # Light green
                elif category == "Overweight":
                    body_color = "#FFD700"  # Gold
                elif category == "Obese Level I":
                    body_color = "#FFA500"  # Orange
                elif category == "Obese Level II":
                    body_color = "#FF6347"  # Tomato
                else:  # Obese Level III
                    body_color = "#DC143C"  # Crimson
                
                # Head
                ax.scatter([0], [2], s=800, c=body_color, alpha=0.8, edgecolors="black", linewidth=3)
                
                # Body style based on BMI
                if bmi < 25:
                    # Normal body
                    ax.plot([0, 0], [1, 1.7], color=body_color, lw=6)
                elif bmi < 30:
                    # Slightly larger body
                    ax.plot([0, 0], [1, 1.7], color=body_color, lw=8)
                else:
                    # Larger body with belly
                    belly_size = min(max((bmi - 20) * 150, 500), 4000)
                    ax.scatter([0], [1.3], s=belly_size, c=body_color, alpha=0.6, edgecolors="black", linewidth=2)
                    ax.plot([0, 0], [1, 1.7], color=body_color, lw=8)
                
                # Arms
                arm_length = 0.8 if bmi < 30 else 1.0
                ax.plot([-arm_length, arm_length], [1.5, 1.5], color=body_color, lw=4)
                
                # Legs
                leg_length = 0.6 if bmi < 30 else 0.7
                ax.plot([0, -leg_length], [1, 0], color=body_color, lw=4)
                ax.plot([0, leg_length], [1, 0], color=body_color, lw=4)
                
                # Set plot limits and style
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-0.5, 3.5)
                ax.axis("off")
                
                # Add title and info
                plt.title(f"👤 {gender} | BMI: {bmi:.1f} ({category})\nBody Type Visualization", 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Add BMI category info
                category_info = {
                    "Underweight": "🔵 BMI < 18.5 - May need to gain healthy weight",
                    "Normal Weight": "🟢 BMI 18.5-24.9 - Healthy weight range",
                    "Overweight": "🟡 BMI 25-29.9 - Consider weight management",
                    "Obese Level I": "🟠 BMI 30-34.9 - Weight loss recommended",
                    "Obese Level II": "🔴 BMI 35-39.9 - Significant weight loss needed",
                    "Obese Level III": "🔴 BMI ≥ 40 - Medical intervention recommended"
                }
                
                # Display the plot
                st.pyplot(fig)
                
                # Add BMI category explanation
                st.markdown("### 📊 **BMI Category Information**")
                st.info(category_info.get(category, "BMI category information"))
                
                # Add health tips based on BMI
                st.markdown("### 💡 **Health Tips for Your BMI Range**")
                
                if category == "Underweight":
                    st.success("""
                    **For Underweight Individuals:**
                    - Focus on healthy weight gain through nutritious foods
                    - Include healthy fats like avocados, nuts, and olive oil
                    - Eat regular meals and snacks throughout the day
                    - Consider working with a healthcare provider
                    """)
                elif category == "Normal Weight":
                    st.success("""
                    **For Normal Weight Individuals:**
                    - Maintain your current healthy lifestyle
                    - Continue balanced eating and regular exercise
                    - Monitor your weight to prevent future changes
                    - Keep up the great work!
                    """)
                elif category == "Overweight":
                    st.warning("""
                    **For Overweight Individuals:**
                    - Focus on gradual, sustainable weight loss
                    - Create a moderate calorie deficit
                    - Increase physical activity gradually
                    - Consider portion control and mindful eating
                    """)
                else:  # Obese categories
                    st.error("""
                    **For Obese Individuals:**
                    - Consult with a healthcare provider for guidance
                    - Focus on significant lifestyle changes
                    - Consider working with a dietitian or nutritionist
                    - Set realistic, achievable weight loss goals
                    - Consider medical supervision for your weight loss journey
                    """)
                
                # Add interactive BMI calculator
                st.markdown("### 🧮 **BMI Calculator**")
                st.info(f"Your current BMI of {bmi:.1f} is in the **{category}** range.")
                
                # Show BMI ranges
                bmi_ranges = {
                    "Underweight": "Below 18.5",
                    "Normal Weight": "18.5 - 24.9",
                    "Overweight": "25.0 - 29.9",
                    "Obese Level I": "30.0 - 34.9",
                    "Obese Level II": "35.0 - 39.9",
                    "Obese Level III": "40.0 and above"
                }
                
                st.markdown("**BMI Categories:**")
                for cat, range_text in bmi_ranges.items():
                    if cat == category:
                        st.markdown(f"- **{cat}**: {range_text} ← **You are here**")
                    else:
                        st.markdown(f"- {cat}: {range_text}")
            else:
                st.error("Please calculate your BMI first by clicking 'Calculate Health Metrics'")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🤖 Powered by Policy Gradient Reinforcement Learning | 
        Built with Streamlit | 
        <a href='https://github.com' target='_blank'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
