function analyzeReview() {
    let review = document.getElementById('reviewText').value;
    if (!review) {
        alert("Please enter a review.");
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: review })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Result: ${data.result}\nAccuracy: ${data.accuracy}%`;
    })
    .catch(error => console.error('Error:', error));
}
