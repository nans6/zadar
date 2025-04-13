from typing import Dict, Any

# --- Configuration: Define Scarcity Thresholds and Risk Levels ---
# Thresholds for identifying grid stress conditions
HIGH_LOAD_THRESHOLD_PCT = 10.0  # Load forecast > 10% above normal
LOW_WIND_THRESHOLD_PCT = -30.0  # Wind forecast < -30% below normal

# Risk levels and corresponding prices
# High Scarcity (High Load AND Low Wind)
HIGH_RISK_PROB = 0.80
HIGH_MAX_PRICE = 1200  # ERCOT prices can spike significantly

# Medium Scarcity (Either High Load OR Low Wind)
MEDIUM_RISK_PROB = 0.45
MEDIUM_MAX_PRICE = 300

# Low Scarcity (Normal conditions)
LOW_RISK_PROB = 0.10
LOW_MAX_PRICE = 75

# Natural Gas price threshold for risk adjustment
HIGH_NG_PRICE_THRESHOLD = 4.0  # $/MMBtu

def assess_price_risk(
    load_prediction: Dict[str, Any],
    renewable_metrics: Dict[str, Any],
    market_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Estimates price spike risk based on forecasted grid scarcity conditions.

    Args:
        load_prediction: Dict with load forecast (e.g., 'load_vs_normal_pct')
        renewable_metrics: Dict with renewable forecast (e.g., 'wind_deviation_from_normal_pct')
        market_data: Optional dict with market prices (e.g., 'nat_gas': {'price': ...})

    Returns:
        Dict containing price_spike_probability (0.0-1.0), expected_max_price ($/MWh),
        and explanatory comment
    """
    # Default to low risk
    probability = LOW_RISK_PROB
    max_price = LOW_MAX_PRICE
    comment = "Low scarcity risk: Normal grid conditions expected."

    # Extract scarcity indicators
    load_dev_pct = load_prediction.get('load_vs_normal_pct')
    wind_dev_pct = renewable_metrics.get('wind_deviation_from_normal_pct')

    # Input validation
    if load_dev_pct is None or wind_dev_pct is None:
        return {
            "price_spike_probability": MEDIUM_RISK_PROB,
            "expected_max_price": MEDIUM_MAX_PRICE,
            "comment": "Missing load or wind deviation data, defaulting to medium risk."
        }

    # Evaluate scarcity conditions
    is_high_load = load_dev_pct > HIGH_LOAD_THRESHOLD_PCT
    is_low_wind = wind_dev_pct < LOW_WIND_THRESHOLD_PCT

    # Determine risk level based on scarcity conditions
    if is_high_load and is_low_wind:
        probability = HIGH_RISK_PROB
        max_price = HIGH_MAX_PRICE
        comment = f"High scarcity risk: High Load ({load_dev_pct:.1f}% above normal) AND Low Wind ({wind_dev_pct:.1f}% below normal)."
    elif is_high_load:
        probability = MEDIUM_RISK_PROB
        max_price = MEDIUM_MAX_PRICE
        comment = f"Medium scarcity risk: High Load ({load_dev_pct:.1f}% above normal)."
    elif is_low_wind:
        probability = MEDIUM_RISK_PROB
        max_price = MEDIUM_MAX_PRICE
        comment = f"Medium scarcity risk: Low Wind ({wind_dev_pct:.1f}% below normal)."

    # Adjust for high natural gas prices if available
    if market_data and 'nat_gas' in market_data:
        ng_price = market_data['nat_gas'].get('price')
        if ng_price and ng_price > HIGH_NG_PRICE_THRESHOLD:
            comment += f" High natural gas price (${ng_price:.2f}) increases risk."
            if probability < HIGH_RISK_PROB:
                probability = min(probability * 1.2, HIGH_RISK_PROB)
                max_price = min(int(max_price * 1.2), HIGH_MAX_PRICE)

    return {
        "price_spike_probability": probability,
        "expected_max_price": max_price,
        "comment": comment
    }

if __name__ == '__main__':
    from pprint import pprint

    # Test cases
    print("\n=== Testing Price Risk Model ===\n")

    # Test 1: High Scarcity
    print("Test 1: High Scarcity (High Load + Low Wind)")
    risk1 = assess_price_risk(
        {'load_vs_normal_pct': 15.0},
        {'wind_deviation_from_normal_pct': -40.0}
    )
    pprint(risk1)

    # Test 2: Medium Scarcity (Low Wind)
    print("\nTest 2: Medium Scarcity (Low Wind Only)")
    risk2 = assess_price_risk(
        {'load_vs_normal_pct': 5.0},
        {'wind_deviation_from_normal_pct': -35.0}
    )
    pprint(risk2)

    # Test 3: Medium Scarcity (High Load)
    print("\nTest 3: Medium Scarcity (High Load Only)")
    risk3 = assess_price_risk(
        {'load_vs_normal_pct': 12.0},
        {'wind_deviation_from_normal_pct': -10.0}
    )
    pprint(risk3)

    # Test 4: Low Scarcity
    print("\nTest 4: Low Scarcity (Normal Conditions)")
    risk4 = assess_price_risk(
        {'load_vs_normal_pct': 2.0},
        {'wind_deviation_from_normal_pct': 5.0}
    )
    pprint(risk4)

    # Test 5: High Scarcity + High Gas Price
    print("\nTest 5: High Scarcity + High Gas Price")
    risk5 = assess_price_risk(
        {'load_vs_normal_pct': 15.0},
        {'wind_deviation_from_normal_pct': -40.0},
        {'nat_gas': {'price': 4.50}}
    )
    pprint(risk5)
