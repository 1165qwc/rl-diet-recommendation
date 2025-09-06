# 🤖 RL Diet Recommendation System

A simplified Reinforcement Learning-powered diet recommendation system built with Streamlit and Policy Gradient algorithms. This system provides personalized diet recommendations based on user health profiles using real obesity data.

## 🌟 Features

- **Policy Gradient RL Agent**: Uses REINFORCE algorithm for personalized recommendations
- **Real Data Integration**: Trained on actual obesity dataset
- **Interactive Web Interface**: Beautiful Streamlit dashboard
- **Health Assessment**: BMI calculation and lifestyle scoring
- **Personalized Recommendations**: 6 different diet strategies
- **Visual Analytics**: Charts and graphs for health insights
- **Easy Deployment**: Ready for Streamlit Cloud deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd rl-diet-recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data file is present**
   - Make sure `processed_obesity_data.csv` is in the project directory

4. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_diet_recommendation.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📊 How to Use

### Step 1: Train the RL Agent
1. Click "🚀 Train Policy Gradient Agent" in the sidebar
2. Wait for training to complete (takes 2-3 minutes)
3. You'll see a training progress chart

### Step 2: Enter Your Health Information
1. **Personal Info**: Height, weight, age, gender
2. **Lifestyle Assessment**: 
   - Vegetable intake (servings/day)
   - Water intake (liters/day)
   - Exercise frequency (days/week)
   - Screen time (hours/day)
   - Main meals per day

### Step 3: Get Your Recommendation
1. Click "🧮 Calculate Health Metrics" to see your BMI and health score
2. Click "🎯 Get Personalized Recommendation" for AI-powered diet advice
3. View detailed food recommendations, tips, and confidence levels

## 🧠 How It Works

### Policy Gradient Algorithm
The system uses a simplified Policy Gradient (REINFORCE) algorithm:

1. **State Space**: 8-dimensional vector representing:
   - BMI (normalized)
   - Age (normalized)
   - Gender (binary)
   - Exercise frequency
   - Water intake
   - Vegetable intake
   - Screen time
   - Meals per day

2. **Action Space**: 6 diet strategies:
   - High Protein, Low Carb
   - Balanced Mediterranean
   - Low Calorie, High Volume
   - Intermittent Fasting
   - Plant-Based Focus
   - Keto-Inspired

3. **Reward Function**: Considers:
   - Action effectiveness
   - BMI compatibility
   - Lifestyle factors
   - Personalization factors

### Training Process
1. Agent learns from real obesity data
2. Simulates user interactions and health outcomes
3. Updates policy based on reward signals
4. Optimizes for personalized recommendations

## 📁 Project Structure

```
rl-diet-recommendation/
├── streamlit_diet_recommendation.py  # Main Streamlit app
├── requirements.txt                 # Python dependencies
├── run_app.py                       # Simple launcher script
├── demo.py                          # Demo script
├── README.md                        # This file
├── DEPLOYMENT_GUIDE.md              # Deployment instructions
└── processed_obesity_data.csv       # Processed dataset (required)
```

## 🔧 Customization

### Adding New Diet Strategies
1. Edit the `actions` dictionary in `DietRecommendationEnvironment`
2. Add corresponding recommendations in `get_diet_recommendations()`
3. Update reward calculation in `_calculate_reward()`

### Modifying the State Space
1. Update `state_dim` in the environment
2. Modify `_normalize_state()` method
3. Update the UI input fields in Streamlit

### Adjusting Training Parameters
1. Change `episodes` in the training call
2. Modify `learning_rate` and `discount_factor` in `PolicyGradientAgent`
3. Adjust `max_steps` in the environment

## 🚀 Deployment

### Local Deployment
```bash
streamlit run streamlit_diet_recommendation.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with the following settings:
   - **Main file path**: `streamlit_diet_recommendation.py`
   - **Python version**: 3.8+

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_diet_recommendation.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 Data Requirements

The system requires the processed obesity data:

- **Required**: `processed_obesity_data.csv` must be in the project directory
- **Data Format**: The file should contain columns: BMI, Age, Gender, ActivityLevel, H2O, eatvege, TechFreq, MainMeal

### Data Format
The processed data should contain these columns:
- `BMI`: Body Mass Index
- `Age`: Age in years
- `Gender`: Male/Female
- `ActivityLevel`: Exercise frequency (0, 3, or 6 days/week)
- `H2O`: Water intake (1-5 liters/day)
- `eatvege`: Vegetable intake (0-5 servings/day)
- `TechFreq`: Screen time (0, 6, or 12 hours/day)
- `MainMeal`: Main meals per day (1-6)

## 🎯 Algorithm Details

### Policy Gradient (REINFORCE)
- **Type**: Policy-based method
- **Advantage**: Direct policy optimization
- **Learning**: Updates policy parameters using gradient ascent
- **Exploration**: Natural exploration through policy sampling

### Reward Design
The reward function considers multiple factors:
- **Base Effectiveness**: Each diet strategy has inherent effectiveness
- **BMI Compatibility**: Rewards strategies suitable for user's BMI range
- **Lifestyle Match**: Considers exercise, diet preferences, and habits
- **Health Goals**: Aligns with weight management objectives

## 🔬 Technical Specifications

- **Framework**: Streamlit
- **RL Algorithm**: Policy Gradient (REINFORCE)
- **State Space**: 8-dimensional continuous
- **Action Space**: 6 discrete actions
- **Training Episodes**: 500 (configurable)
- **State Normalization**: Min-max scaling to [0,1]
- **Policy Network**: Linear model with softmax output

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install dependencies with `pip install -r requirements.txt`
2. **Data not found**: Run `python preprocess_data.py` to generate synthetic data
3. **Training takes too long**: Reduce episodes in the training call
4. **Memory issues**: Reduce the dataset size or training episodes

### Performance Tips

- Use a GPU for faster training (if available)
- Reduce training episodes for quicker testing
- Cache the trained model for faster subsequent runs

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

## 🎉 Acknowledgments

- Streamlit team for the amazing framework
- OpenAI for RL algorithm inspiration
- The obesity dataset contributors
- The open-source community

---

**Built with ❤️ using Streamlit and Policy Gradient RL**
