def get_param_severity(value, param):
    """
    Classifies a parameter's value into severity levels.
    """
    if param == "amh":
        if value > 8:
            return "Severe"
        elif 5 <= value <= 8:
            return "Moderate"
        else:
            return "Mild"
    elif param in ["beta_hcg_1", "beta_hcg_2"]:
        if value > 10:
            return "Severe"
        elif 5 < value <= 10:
            return "Moderate"
        else:
            return "Mild"
    elif param == "bmi":
        if value > 30:
            return "Severe"
        elif 25 <= value <= 30:
            return "Moderate"
        else:
            return "Mild"
    elif param == "cycle_length":
        if value > 40:
            return "Severe"
        elif 35 <= value <= 40:
            return "Moderate"
        else:
            return "Mild"
    elif param == "age":
        if value > 35:
            return "Severe"
        elif 28 <= value <= 35:
            return "Moderate"
        else:
            return "Mild"
    else:
        return "Mild"


def get_recommendations_by_param(amh, beta_hcg_1, beta_hcg_2, bmi, age, cycle_length, severity):
    recommendations = {}  # Initialize an empty dictionary to store recommendations

    # Compute severity levels for each parameter
    amh_sev = get_param_severity(amh, "amh")
    beta1_sev = get_param_severity(beta_hcg_1, "beta_hcg_1")
    beta2_sev = get_param_severity(beta_hcg_2, "beta_hcg_2")
    bmi_sev = get_param_severity(bmi, "bmi")
    age_sev = get_param_severity(age, "age")
    cycle_sev = get_param_severity(cycle_length, "cycle_length")

    # AMH
    if amh_sev == "Severe":
        recommendations["AMH"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">- Strict low-GI diet with high nutrient density.</p>
    <p style="font-size: 18px; color: #333;">- <b>Non-starchy vegetables:</b> Unlimited amounts of broccoli, cauliflower, spinach, bell peppers, and asparagus.</p>
    <p style="font-size: 18px; color: #333;">- <b>Lean protein:</b> 4 – 6 oz of chicken breast, fish, turkey, or tofu at each meal.</p>
    <p style="font-size: 18px; color: #333;">- <b>Healthy fats:</b> Small amounts of avocado, olive oil, and nuts.</p>
    <p style="font-size: 18px; color: #333;">- <b>Limit carbohydrates:</b> Very small amounts of low-GI carbs like 1/4 cup quinoa or sweet potato.</p>
    <p style="font-size: 18px; color: #333;">- <b>Example meals:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Breakfast: Vegetable omelet.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Lunch: Large salad with grilled chicken.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Dinner: Baked fish with steamed vegetables.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Snacks: Nuts or non-starchy vegetables.</p>
    <p style="font-size: 18px; color: #333;">- Supplement with Vitamin D and Omega-3s (consult doctor).</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - <b>Low-impact cardio</b>: 30 mins of walking or swimming daily.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Gentle strength training:</b> 2–3x/week (light weights/bodyweight).</p>
    <p style="font-size: 18px; color: #333;">    - Stress-reducing activities like yoga or meditation.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - 30 mins of daily meditation or deep breathing.</p>
    <p style="font-size: 18px; color: #333;">    - Regular yoga or tai chi sessions.</p>
    <p style="font-size: 18px; color: #333;">    - Prioritize 8 hours of quality sleep.</p>
    <p style="font-size: 18px; color: #333;">    - Therapy/counseling for emotional well-being.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Immediate consultation with a fertility specialist.</p>
    <p style="font-size: 18px; color: #333;">    - Medical interventions like fertility meds/IVF may be necessary.</p>
    <p style="font-size: 18px; color: #333;">    - Regular monitoring by a doctor.</p>
</div>
"""
    elif amh_sev == "Moderate":
        recommendations["AMH"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">    - <b>Berries</b>: Daily smoothie with 1 cup mixed berries and 1 tbsp chia seeds.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Leafy Greens:</b> Large salad with spinach, kale, and colorful veggies.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Fatty Fish:</b> Salmon/mackerel (3-4 oz) 2-3 times/week.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Nuts and Seeds:</b> Soaked almonds, walnuts, ground flaxseeds.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Olive Oil:</b> 2-3 tbsp/day for cooking or dressing.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><strong>Example Meals:</strong></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Breakfast:</b> Overnight oats with berries and nuts.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Lunch:</b> Grilled salmon salad with avocado.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Dinner:</b> Chicken stir-fry with quinoa and veggies.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Snacks:</b> Greek yogurt with berries, or almonds.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXCERISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Brisk walking (120 steps/min), cycling (30–45 mins, 80–90 RPM), swimming.</p>
    <p style="font-size: 18px; color: #333;">    - Yoga (Hatha/Vinyasa), Pilates for core and flexibility.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Meditation (15–20 mins daily) using apps like Calm.</p>
    <p style="font-size: 18px; color: #333;">    - Deep breathing (Box method) 5–10 mins.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Sleep:</b> Fixed schedule, relaxing routine, cool dark room.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Reproductive endocrinologist consultation.</p>
    <p style="font-size: 18px; color: #333;">    - Discuss CoQ10 and Vitamin D supplementation.</p>
</div>
"""
    else:
        recommendations["AMH"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">AMH level is good. Continue with nutritious food and regular checkups.</p>
</div>
"""

    # Beta-HCG 1
    if beta1_sev == "Severe":
        recommendations["Beta-HCG_1"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Personalized meal plans with a registered dietitian.</p>
    <p style="font-size: 18px; color: #333;">    - Very low carbohydrate intake, high protein and healthy fat intake.</p>
    <p style="font-size: 18px; color: #333;">    - Focus on non-starchy vegetables, lean meats, and healthy fats.</p>
    <p style="font-size: 18px; color: #333;">    - Hydration: Drink at least 10 glasses of water daily.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>Insulin Resistance Management:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Incorporate insulin-sensitizing foods and supplements (under medical supervision).</p>
    <p style="font-size: 18px; color: #333;">    - Careful calorie restriction to manage weight gain and improve insulin sensitivity.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXCERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Gentle Movement Only: Short walks as tolerated, with close monitoring of symptoms.</p>
    <p style="font-size: 18px; color: #333;">    - Prioritize rest and avoid strenuous activity.</p>
    <p style="font-size: 18px; color: #333;">    - All exercise must be approved by the doctor.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Regular therapy/counseling sessions.</p>
    <p style="font-size: 18px; color: #333;">    - Daily practice of relaxation techniques (progressive muscle relaxation, guided imagery).</p>
    <p style="font-size: 18px; color: #333;">    - Join support groups for emotional support.</p>
    <p style="font-size: 18px; color: #333;">    - Sleep Optimization: 8–9 hours of quality sleep nightly.</p>
</div>
"""
    elif beta1_sev == "Moderate":
        recommendations["Beta-HCG_1"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>LOW-GI EMPHASIS:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Whole, unprocessed foods (GI ≤ 55).</p>
    <p style="font-size: 18px; color: #333;">    - <b>Complex carbohydrates:</b> Quinoa (1/2 cup), sweet potatoes (1/2 cup), brown rice (1/2 cup).</p>
    <p style="font-size: 18px; color: #333;">    - <b>Lean proteins:</b> Chicken/fish/turkey (4 oz), lentils (1 cup).</p>
    <p style="font-size: 18px; color: #333;">    - <b>Healthy fats:</b> Avocado (1/4), olive oil (2 tbsp), nuts/seeds.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><strong>Example Meals:</strong></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Breakfast:</b> Oatmeal with berries and walnuts.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Lunch:</b> Salad with grilled chicken, avocado, and non-starchy vegetables.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Dinner:</b> Baked salmon with steamed asparagus and quinoa.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>Functional Foods:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Cinnamon (1/2–1 tsp), apple cider vinegar (1–2 tbsp diluted), chromium-rich foods.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>Anti-inflammatory Diet:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Fatty fish, turmeric, ginger.</p>

    <p style="font-size: 22px; color: #333;"><strong>Hydration:</strong> At least 8 glasses/day.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - 30 mins of moderate cardio (walking, swimming, cycling).</p>
    <p style="font-size: 18px; color: #333;">    - Gentle strength training, yoga or Pilates.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Daily mindfulness meditation (15–20 mins).</p>
    <p style="font-size: 18px; color: #333;">    - Breathing exercises.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Sleep:</b> 7–8 hours with consistent bedtime routine.</p>
    <p style="font-size: 18px; color: #333;">    - <b>Emotional Support:</b> Therapy and PCOS support groups.</p>
</div>
"""
    else:
        recommendations["Beta-HCG_1"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">Beta-HCG level is within a healthy range. Maintain regular sleep and balanced nutrition.</p>
</div>
"""

    # Beta-HCG 2
    if beta2_sev == "Severe":
        recommendations["Beta-HCG_2"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>PCOS-Focused Nutrition:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Low-GI diet with personalized plans and calorie restriction.</p>
    <p style="font-size: 18px; color: #333;">    - Monitor blood sugar and weight regularly.</p>
    <p style="font-size: 18px; color: #333;">    - Drink at least 10 glasses of water daily.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Only gentle walking as tolerated.</p>
    <p style="font-size: 18px; color: #333;">    - Avoid strenuous activity.</p>
    <p style="font-size: 18px; color: #333;">    - Follow doctor's recommendations strictly.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Regular therapy or counseling sessions.</p>
    <p style="font-size: 18px; color: #333;">    - Daily relaxation (deep breathing, guided imagery).</p>
    <p style="font-size: 18px; color: #333;">    - Support from groups and family.</p>
    <p style="font-size: 18px; color: #333;">    - Sleep: Aim for 8–9 hours of restful sleep.</p>
</div>
"""
    elif beta2_sev == "Moderate":
        recommendations["Beta-HCG_2"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>Balanced Diet:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Maintain a low-GI diet with portion control.</p>
    <p style="font-size: 18px; color: #333;">    - Prioritize lean protein and nutrient-dense foods.</p>
    <p style="font-size: 18px; color: #333;">    - Drink at least 8 glasses of water daily.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Gentle walking or low-impact aerobics.</p>
    <p style="font-size: 18px; color: #333;">    - Avoid strenuous activity.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Continue relaxation techniques.</p>
    <p style="font-size: 18px; color: #333;">    - Emotional support is important.</p>
    <p style="font-size: 18px; color: #333;">    - Prioritize regular sleep for hormonal balance.</p>
</div>
"""
    else:
        recommendations["Beta-HCG_2"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">Beta-HCG level is within a healthy range. Continue balanced nutrition and sleep habits.</p>
</div>
"""

    # BMI
    if bmi_sev == "Severe":
        recommendations["BMI"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">- Structured Meal Plan (Dietitian):</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Work with a registered dietitian to create a personalized meal plan.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Prioritize whole, unprocessed foods.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Consider a low-carbohydrate approach under medical supervision.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Increase water intake (at least 8–10 glasses per day).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Increase lean protein intake to promote satiety.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">- Gradual Increase:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Start with low-impact activities like walking or water aerobics.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Gradually increase duration and intensity as fitness improves.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Work with a personal trainer experienced in weight management.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">- Consistent Stress Reduction:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Practice stress-reduction techniques daily (meditation, deep breathing).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Prioritize sleep and rest.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;">- Medical Weight Loss Options:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Discuss medical weight loss options with a doctor (medications, bariatric surgery).</p>
    <p style="font-size: 18px; color: #333;">- Consistent Monitoring:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Regular checkups with a doctor and dietitian.</p>
</div>
"""

    elif bmi_sev == "Moderate":
        recommendations["BMI"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">- Detailed Portion Control:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Use smaller plates (9-inch diameter).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Measure food portions using measuring cups and spoons.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Focus on lean proteins (chicken breast, fish) and non-starchy vegetables (broccoli, spinach).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Increase fiber intake: Add chia seeds to smoothies, flaxseeds to yogurt, and beans to soups and salads.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Replace sugary drinks with water or herbal tea.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;">- Example Meal Plan:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Breakfast:</b> Greek yogurt (1 cup) with berries (1/2 cup) and a sprinkle of chia seeds (1 tablespoon).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Lunch:</b> Large salad with grilled chicken (4 oz), mixed greens, and a light vinaigrette dressing.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Dinner:</b> Baked salmon (4 oz) with steamed vegetables (broccoli, carrots, asparagus).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - <b>Snacks:</b> Apple slices with almond butter (2 tablespoons) or a handful of almonds.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Structured Cardio:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Brisk walking, jogging, cycling, or swimming for 45–60 minutes, 5–6 days per week.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Use a fitness tracker to monitor heart rate and intensity.</p>

    <p style="font-size: 18px; color: #333;"><b>- Strength Training:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Include full-body strength training exercises (squats, lunges, push-ups) 2–3 times per week.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Use weights or resistance bands.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Consistent Stress Reduction:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Practice mindfulness meditation for 10–15 minutes daily.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Engage in relaxing activities like reading, listening to music, or taking a warm bath.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Prioritize 7–8 hours of quality sleep.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Food Journaling:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Track food intake for several weeks to identify patterns and adjust portions.</p>
    <p style="font-size: 18px; color: #333;"><b>- Regular Weight Monitoring:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Weigh yourself weekly to monitor progress.</p>
</div>
"""
    else:
        recommendations["BMI"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">Healthy weight. Continue with consistent physical activity.</p>
</div>
"""

    # Menstrual Cycle
    if cycle_sev == "Severe":
        recommendations["Menstrual Cycle"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>PCOS-Focused Nutrition:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Intensive Hormonal Regulation Diet:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Work with a registered dietitian to create a personalized meal plan focused on strict low-GI principles and hormonal balance.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Prioritize whole, unprocessed foods, lean proteins, and healthy fats.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Limit refined carbohydrates, sugary foods, and processed foods.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Consider a low-carbohydrate approach under medical supervision.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Example Meal Plan: Similar to the moderate plan but with stricter portion control and fewer carbohydrates.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><b>- Insulin Resistance Management:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Incorporate insulin-sensitizing foods and supplements (under medical supervision).</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><b>- Anti-inflammatory Foods:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Focus on a diet rich in anti-inflammatory foods to reduce inflammation and support hormonal balance.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">- Gentle Exercise:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Engage in low-impact exercise (walking, swimming) as tolerated, with close monitoring of symptoms.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Avoid high-intensity exercise which may worsen hormonal imbalance.</p>

    <p style="font-size: 10px; color: #333; margin-bottom: 10px;"><b>- Stress Reduction Exercises:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Prioritize yoga or tai chi for stress relief and hormone regulation.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Intensive Stress Reduction:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Practice relaxation techniques (meditation, deep breathing) daily.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Consider therapy or counseling to support emotional well-being.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>Sleep Optimization:</strong></p>
    <p style="font-size: 18px; color: #333;">- Prioritize 8–9 hours of quality sleep per night to support hormone balance and insulin sensitivity.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Medical Intervention:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Work with a gynecologist or endocrinologist on hormonal treatments if necessary.</p>

    <p style="font-size: 18px; color: #333;"><b>- Cycle Tracking:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Track basal body temperature and ovulation to monitor fluctuations.</p>

    <p style="font-size: 18px; color: #333;"><b>- Limit Exposure to Endocrine Disruptors:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Minimize exposure to environmental toxins that affect hormones.</p>
</div>
"""
    elif cycle_sev == "Moderate":
        recommendations["Menstrual Cycle"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>PCOS-Focused Nutrition:</strong></p>
    <p style="font-size: 18px; color: #333;">- Hormonal Balance Diet:</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Follow a low-GI diet to stabilize blood sugar and support hormonal regulation.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Increase fiber (30g/day) from whole grains, legumes, and vegetables.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Include healthy fats (avocados, nuts, seeds, olive oil).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Limit processed foods, sugary drinks, and excess caffeine.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><b>- Example Meal Plan:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Breakfast: Oatmeal with berries and flaxseeds.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Lunch: Large salad with grilled chicken or fish and avocado.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Dinner: Lentil soup with whole-grain bread.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Snacks: Apple slices with almond butter or a handful of mixed nuts.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><b>- Insulin Sensitivity Support:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Use cinnamon (1 tsp/day) and diluted apple cider vinegar (1–2 tbsp).</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Include chromium-rich foods like broccoli and grapes.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><b>- Anti-inflammatory Foods:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Eat foods rich in omega-3s (fatty fish, flaxseeds, chia seeds), turmeric, and ginger.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Moderate-Intensity Cardio:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Do 30–45 minutes of brisk walking, cycling, or swimming 4–5 times/week.</p>

    <p style="font-size: 18px; color: #333;"><b>- Strength Training:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Include 2–3 sessions/week of exercises like squats, lunges, and resistance training.</p>

    <p style="font-size: 18px; color: #333;"><b>- Stress Reduction Exercises:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Practice yoga or Pilates 2–3 times/week.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Mindfulness and Meditation:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Meditate for 15–20 minutes daily to lower cortisol.</p>

    <p style="font-size: 18px; color: #333;"><b>- Deep Breathing:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Do diaphragmatic breathing for 5–10 minutes, several times daily.</p>

    <p style="font-size: 18px; color: #333;"><b>-Sleep Hygiene:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Get 7–8 hours of quality sleep every night.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Maintain a consistent sleep schedule and calming bedtime routine.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHNAGES:</strong></p>
    <p style="font-size: 18px; color: #333;"><b>- Cycle Tracking:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Track menstrual cycles to identify trends and manage PCOS better.</p>

    <p style="font-size: 18px; color: #333;"><b>- Acupuncture:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Consider it for cycle regulation and hormonal support.</p>

    <p style="font-size: 18px; color: #333;"><b>- Limit Exposure to Endocrine Disruptors:</b></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Avoid BPA, phthalates, and other hormone-disrupting chemicals.</p>
</div>
"""
    else:
        recommendations["Menstrual Cycle"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">Regular cycles. Keep following a balanced lifestyle.</p>
</div>
"""

    if age_sev == "Severe":
        recommendations["Age"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">     - Focus on cardiovascular and metabolic health.</p>
    <p style="font-size: 18px; color: #333;">  <b>- High-Fiber Diet</b>: Whole grains, legumes, fruits, vegetables.</p>
    <p style="font-size: 18px; color: #333;">  <b>- Heart-Healthy Fats:</b> Olive oil, avocados, nuts, seeds.</p>
    <p style="font-size: 18px; color: #333;">  <b>- Calcium & Vitamin D:</b> Support bone health (with medical advice).</p>
    <p style="font-size: 18px; color: #333;">     - Limit sodium, red meat, and saturated fats.</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><strong><b>Example Meal Plan:</b></strong></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Breakfast: Oatmeal with berries and walnuts.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Lunch: Salad with grilled fish/chicken and avocado.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Dinner: Baked chicken with roasted vegetables.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Snacks: Almonds or fruit.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Low-Impact Cardio: Walking, swimming, water aerobics.</p>
    <p style="font-size: 18px; color: #333;">    - Strength Training: 2-3 sessions/week.</p>
    <p style="font-size: 18px; color: #333;">    - Flexibility & Balance: Yoga or tai chi.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Daily mindfulness and meditation.</p>
    <p style="font-size: 18px; color: #333;">    - 7-8 hours of quality sleep.</p>
    <p style="font-size: 18px; color: #333;">    - Maintain strong social connections.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Regular health screenings (diabetes, heart, etc.).</p>
    <p style="font-size: 18px; color: #333;">    - Discuss HRT if relevant.</p>
    <p style="font-size: 18px; color: #333;">    - Routine checkups with specialists.</p>
</div>
"""
    elif age_sev == "Moderate":
        recommendations["Age"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>FOOD:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Focus on fertility and hormonal balance.</p>
    <p style="font-size: 18px; color: #333;">    - Folate-Rich Foods: Spinach, lentils, kale (400 mcg/day).</p>
    <p style="font-size: 18px; color: #333;">    - Vitamin D: Fish, egg yolks, fortified foods (or supplement).</p>
    <p style="font-size: 18px; color: #333;">    - Antioxidants: Berries, dark chocolate (in moderation).</p>
    <p style="font-size: 18px; color: #333;">    - Limit inflammatory foods (processed foods, red meat).</p>

    <p style="font-size: 18px; color: #333; margin-bottom: 10px;"><strong>Example Meal Plan:</strong></p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Breakfast: Smoothie with spinach, berries, almond milk, eggs.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Lunch: Salad with grilled chicken or fish.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Dinner: Lentil soup or baked salmon with roasted veggies.</p>
    <p style="font-size: 18px; color: #333; margin-left: 20px;">    - Snacks: Almonds or small dark chocolate.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>EXERCISE:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Cardio: 30-45 mins, 4-5 days/week.</p>
    <p style="font-size: 18px; color: #333;">    - Strength Training: 2-3 times/week.</p>
    <p style="font-size: 18px; color: #333;">    - Yoga or Pilates for stress relief.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>STRESS MANAGEMENT:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Daily meditation (10–15 mins).</p>
    <p style="font-size: 18px; color: #333;">    - Sleep hygiene and consistent bedtime.</p>
    <p style="font-size: 18px; color: #333;">    - Relaxing hobbies like reading or music.</p>

    <p style="font-size: 22px; color: #333; margin-bottom: 10px;"><strong>OTHER LIFESTYLE CHANGES:</strong></p>
    <p style="font-size: 18px; color: #333;">    - Fertility Planning (if needed): Preconception counseling.</p>
    <p style="font-size: 18px; color: #333;">    - Regular medical checkups for hormone and insulin monitoring.</p>
</div>
"""
    else:
        recommendations["Age"] = """
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <p style="font-size: 18px; color: #333;">Young age. Great time to prevent PCOS progression with consistent care.</p>
</div>
"""

    return recommendations

# Example usage
if __name__ == "__main__":
    recs = get_recommendations_by_param(
        amh=3.2,
        beta_hcg_1=1.7,
        beta_hcg_2=2.5,
        bmi=22.5,
        age=24,
        cycle_length=30,
        severity='Mild'
    )
    for r in recs:
        print(f"**{r}**:")
        print(recs[r]) #Print the multiline string and let python handle the \n
        print()