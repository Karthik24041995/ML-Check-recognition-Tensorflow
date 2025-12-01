"""
Amount Validation Module for Check Recognition
Validates recognized amounts for reasonableness and accuracy
"""

import numpy as np
import re
from datetime import datetime


class AmountValidator:
    """Validates recognized check amounts"""
    
    def __init__(self, min_amount=0.01, max_amount=100000.00,
                 min_confidence=0.7, currency='USD'):
        """
        Initialize validator with business rules
        
        Args:
            min_amount: Minimum valid check amount
            max_amount: Maximum valid check amount
            min_confidence: Minimum confidence threshold (0-1)
            currency: Currency code ('USD', 'INR', 'EUR', etc.)
        """
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.min_confidence = min_confidence
        self.currency = currency
        self.validation_errors = []
    
    def validate_digit_confidence(self, predictions, confidences):
        """
        Validate that digit predictions have sufficient confidence
        
        Args:
            predictions: List of predicted digits
            confidences: List of confidence scores (0-1)
        
        Returns:
            (is_valid, avg_confidence, low_confidence_digits)
        """
        if not predictions or not confidences:
            return False, 0.0, []
        
        avg_confidence = np.mean(confidences)
        low_conf_digits = [
            (idx, digit, conf)
            for idx, (digit, conf) in enumerate(zip(predictions, confidences))
            if conf < self.min_confidence
        ]
        
        is_valid = len(low_conf_digits) == 0 and avg_confidence >= self.min_confidence
        
        if not is_valid:
            self.validation_errors.append(
                f"Low confidence: avg={avg_confidence:.2f}, "
                f"{len(low_conf_digits)} digits below threshold"
            )
        
        return is_valid, avg_confidence, low_conf_digits
    
    def validate_amount_range(self, amount):
        """
        Validate that amount is within reasonable range
        
        Args:
            amount: Numeric amount
        
        Returns:
            is_valid (bool)
        """
        is_valid = self.min_amount <= amount <= self.max_amount
        
        if not is_valid:
            self.validation_errors.append(
                f"Amount ${amount:,.2f} outside valid range "
                f"[${self.min_amount:,.2f}, ${self.max_amount:,.2f}]"
            )
        
        return is_valid
    
    def validate_amount_format(self, amount_str):
        """
        Validate amount format (should be digits only or with decimal)
        
        Args:
            amount_str: String representation of amount
        
        Returns:
            (is_valid, formatted_amount)
        """
        # Remove common formatting characters
        cleaned = amount_str.replace(',', '').replace('$', '').strip()
        
        # Check if valid number format
        pattern = r'^\d+(\.\d{1,2})?$'
        is_valid = bool(re.match(pattern, cleaned))
        
        if is_valid:
            try:
                formatted_amount = float(cleaned)
                return True, formatted_amount
            except ValueError:
                is_valid = False
        
        if not is_valid:
            self.validation_errors.append(
                f"Invalid amount format: '{amount_str}'"
            )
            return False, None
        
        return is_valid, None
    
    def detect_anomalies(self, digits, confidences):
        """
        Detect potential anomalies in recognition
        
        Args:
            digits: List of recognized digits
            confidences: List of confidence scores
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for repeated digits (might indicate error)
        if len(digits) >= 3:
            for i in range(len(digits) - 2):
                if digits[i] == digits[i+1] == digits[i+2]:
                    anomalies.append({
                        'type': 'repeated_digits',
                        'position': i,
                        'digit': digits[i],
                        'message': f'Three consecutive {digits[i]}s at position {i}'
                    })
        
        # Check for low confidence outliers
        if confidences:
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            for idx, conf in enumerate(confidences):
                if conf < mean_conf - 2 * std_conf:
                    anomalies.append({
                        'type': 'confidence_outlier',
                        'position': idx,
                        'confidence': conf,
                        'message': f'Digit at position {idx} has unusually low confidence: {conf:.2f}'
                    })
        
        # Check for unusual amount patterns (all zeros, all nines, etc.)
        if len(set(digits)) == 1:
            anomalies.append({
                'type': 'uniform_digits',
                'digit': digits[0],
                'message': f'All digits are {digits[0]} - unusual pattern'
            })
        
        return anomalies
    
    def validate_complete(self, predictions, confidences, amount_str=None):
        """
        Complete validation pipeline
        
        Args:
            predictions: List of predicted digits
            confidences: List of confidence scores
            amount_str: Optional string representation of amount
        
        Returns:
            Dictionary with validation results
        """
        self.validation_errors = []
        
        # Validate digit confidence
        conf_valid, avg_conf, low_conf = self.validate_digit_confidence(
            predictions, confidences
        )
        
        # Build amount string if not provided
        if amount_str is None:
            amount_str = ''.join(map(str, predictions))
        
        # Insert decimal point (assume last 2 digits are cents)
        if len(amount_str) > 2:
            amount_formatted = amount_str[:-2] + '.' + amount_str[-2:]
        else:
            amount_formatted = '0.' + amount_str.zfill(2)
        
        # Validate format
        format_valid, amount_value = self.validate_amount_format(amount_formatted)
        
        # Validate range
        range_valid = False
        if format_valid and amount_value is not None:
            range_valid = self.validate_amount_range(amount_value)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(predictions, confidences)
        
        # Overall validation
        is_valid = conf_valid and format_valid and range_valid
        
        # Format with correct currency symbol
        currency_symbol = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£'}.get(self.currency, '$')
        
        return {
            'is_valid': is_valid,
            'amount': amount_value if format_valid else None,
            'amount_formatted': f"{currency_symbol}{amount_value:,.2f}" if format_valid and amount_value else amount_str,
            'confidence': {
                'average': avg_conf,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'low_confidence_digits': low_conf
            },
            'validations': {
                'confidence_valid': conf_valid,
                'format_valid': format_valid,
                'range_valid': range_valid
            },
            'anomalies': anomalies,
            'errors': self.validation_errors,
            'raw_digits': predictions,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_validation_summary(self, validation_result):
        """
        Get human-readable validation summary
        
        Args:
            validation_result: Result from validate_complete()
        
        Returns:
            String summary
        """
        summary = []
        
        if validation_result['is_valid']:
            summary.append("✓ Validation PASSED")
            summary.append(f"  Amount: {validation_result['amount_formatted']}")
            summary.append(f"  Average Confidence: {validation_result['confidence']['average']:.2%}")
        else:
            summary.append("✗ Validation FAILED")
            summary.append(f"  Detected Amount: {validation_result['amount_formatted']}")
            
            if validation_result['errors']:
                summary.append("  Errors:")
                for error in validation_result['errors']:
                    summary.append(f"    - {error}")
        
        if validation_result['anomalies']:
            summary.append("  Anomalies Detected:")
            for anomaly in validation_result['anomalies']:
                summary.append(f"    - {anomaly['message']}")
        
        return '\n'.join(summary)


class CheckValidator(AmountValidator):
    """Extended validator with check-specific rules"""
    
    def __init__(self, min_amount=0.01, max_amount=100000.00,
                 min_confidence=0.7, check_date=None):
        super().__init__(min_amount, max_amount, min_confidence)
        self.check_date = check_date
    
    def validate_check_date(self, check_date=None):
        """
        Validate check date (not post-dated, not too old)
        
        Args:
            check_date: datetime object or None
        
        Returns:
            is_valid (bool)
        """
        if check_date is None:
            check_date = self.check_date
        
        if check_date is None:
            return True  # Skip if no date provided
        
        today = datetime.now()
        
        # Check if post-dated (future date)
        if check_date > today:
            self.validation_errors.append(
                f"Check is post-dated: {check_date.strftime('%Y-%m-%d')}"
            )
            return False
        
        # Check if too old (e.g., more than 6 months)
        days_old = (today - check_date).days
        if days_old > 180:
            self.validation_errors.append(
                f"Check is {days_old} days old (may be stale)"
            )
            return False
        
        return True
    
    def validate_payee_present(self, payee_name):
        """Validate that payee name is present"""
        if not payee_name or len(payee_name.strip()) < 2:
            self.validation_errors.append("Payee name missing or invalid")
            return False
        return True
    
    def validate_signature_present(self, has_signature):
        """Validate that signature is present"""
        if not has_signature:
            self.validation_errors.append("Signature not detected")
            return False
        return True


def validate_amount(predictions, confidences, min_confidence=0.7):
    """
    Convenience function to validate recognized amount
    
    Args:
        predictions: List of predicted digits
        confidences: List of confidence scores
        min_confidence: Minimum acceptable confidence
    
    Returns:
        Validation result dictionary
    """
    validator = AmountValidator(min_confidence=min_confidence)
    result = validator.validate_complete(predictions, confidences)
    return result


# Example usage
if __name__ == "__main__":
    print("Amount Validation Module")
    print("=" * 60)
    print("This module validates recognized check amounts:")
    print("- Confidence validation")
    print("- Amount range validation")
    print("- Format validation")
    print("- Anomaly detection")
    print("\nUsage example:")
    print("  from amount_validator import validate_amount")
    print("  ")
    print("  predictions = [1, 2, 3, 4, 5]")
    print("  confidences = [0.95, 0.88, 0.92, 0.85, 0.90]")
    print("  result = validate_amount(predictions, confidences)")
    print("  print(result['amount_formatted'])  # $123.45")
    print("=" * 60)
    
    # Demo validation
    print("\nDemo 1: Valid amount")
    demo_predictions_1 = [1, 2, 3, 4, 5]
    demo_confidences_1 = [0.95, 0.88, 0.92, 0.85, 0.90]
    
    validator = AmountValidator()
    result_1 = validator.validate_complete(demo_predictions_1, demo_confidences_1)
    print(validator.get_validation_summary(result_1))
    
    print("\n" + "=" * 60)
    print("Demo 2: Low confidence amount")
    demo_predictions_2 = [9, 9, 9, 9, 9]
    demo_confidences_2 = [0.95, 0.45, 0.92, 0.55, 0.90]
    
    result_2 = validator.validate_complete(demo_predictions_2, demo_confidences_2)
    print(validator.get_validation_summary(result_2))
    
    print("\n" + "=" * 60)
    print("Demo 3: Out of range amount")
    demo_predictions_3 = [9, 9, 9, 9, 9, 9, 9, 9]
    demo_confidences_3 = [0.95] * 8
    
    result_3 = validator.validate_complete(demo_predictions_3, demo_confidences_3)
    print(validator.get_validation_summary(result_3))
