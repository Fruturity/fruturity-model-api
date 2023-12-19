# Fruturity Model API Documentation
## Overview

The Fruturity Model API provides access to detect the ripeness of the fruit with CNN image classification.

### Base Url
```link
https://fruturity.com
```

### Technology
This API is built with Flask, a framework for Python

## Endpoints
### Predict Fruit Image
- **Endpoint**: `/prediction`
- **Method**: `POST`
- **Description**: Predict the ripeness of the fruit.
- **Request Parameters**: None
- **Body Parameters**:
    - `image` (file, required): image(.jpg, .png, .jpeg) that will be predicted.
- **Request**:
    - Content-Type: application/json
    - Body:
    ```json
    {
        "image": [image file]
    }
    ```
- **Response**:
    - Status code: `200 OK`
    - Body:
    ```json
    {
        "data": {
            "fruit_types_prediction": "Ripe"
        },
        "status": {
            "code": 200,
            "message": "Prediction success"
        }
    }
    ```

