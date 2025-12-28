import numpy as np

def predict_remaining_cgpa(completed_data):
    """
    Predicts missing semester CGPAs using Degree-2 Polynomial Regression.
    
    Args:
        completed_data: List of dicts [{"semester": 1, "cgpa": 7.8}, ...]
        
    Returns:
        List of dicts [{"semester": 6, "predicted_cgpa": 8.5}, ...]
    """
    if len(completed_data) < 2:
        return []

    semesters = np.array([d["semester"] for d in completed_data])
    cgpas = np.array([d["cgpa"] for d in completed_data])

    # Degree-1 Linear Regression (More stable for academic predictions)
    degree = 1
    coefficients = np.polyfit(semesters, cgpas, degree)
    poly = np.poly1d(coefficients)

    last_sem = int(max(semesters))
    predictions = []

    for sem in range(last_sem + 1, 9):
        predicted = poly(sem)
        predicted = round(float(predicted), 2)

        # Clamp CGPA to valid range
        predicted = max(0.0, min(10.0, predicted))

        predictions.append({
            "semester": sem,
            "predicted_cgpa": predicted
        })

    return predictions
