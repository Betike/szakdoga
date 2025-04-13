import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Card, Spinner, Alert } from 'react-bootstrap';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PredictionPage = () => {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [modelType, setModelType] = useState('team-based');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  // In a real app, this would come from an API call
  useEffect(() => {
    // Simulated teams data - in a real app, this would be fetched from the backend
    const sampleTeams = [
      { id: 1, name: 'Manchester United' },
      { id: 2, name: 'Liverpool FC' },
      { id: 3, name: 'FC Barcelona' },
      { id: 4, name: 'Real Madrid' },
      { id: 5, name: 'Bayern Munich' },
      { id: 6, name: 'Juventus' },
      { id: 7, name: 'Paris Saint-Germain' },
      { id: 8, name: 'Manchester City' },
      { id: 9, name: 'Chelsea FC' },
      { id: 10, name: 'Arsenal' }
    ];
    setTeams(sampleTeams);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!homeTeam || !awayTeam) {
      setError('Please select both teams');
      return;
    }
    
    if (homeTeam === awayTeam) {
      setError('Please select different teams');
      return;
    }
    
    setError('');
    setLoading(true);
    
    // In a real app, this would be an API call to your backend
    // Here, we're simulating the prediction with a timeout
    setTimeout(() => {
      const samplePrediction = {
        homeWin: Math.random() * 0.6 + 0.2, // Between 0.2 and 0.8
        draw: Math.random() * 0.4, // Between 0 and 0.4
        awayWin: Math.random() * 0.4 // Between 0 and 0.4
      };
      
      // Normalize to ensure they sum to 1
      const total = samplePrediction.homeWin + samplePrediction.draw + samplePrediction.awayWin;
      samplePrediction.homeWin = samplePrediction.homeWin / total;
      samplePrediction.draw = samplePrediction.draw / total;
      samplePrediction.awayWin = samplePrediction.awayWin / total;
      
      setPrediction(samplePrediction);
      setLoading(false);
    }, 1500); // Simulate loading
  };

  // Prepare chart data for the prediction
  const getPredictionChart = () => {
    if (!prediction) return null;
    
    const data = {
      labels: ['Home Win', 'Draw', 'Away Win'],
      datasets: [
        {
          data: [
            (prediction.homeWin * 100).toFixed(1),
            (prediction.draw * 100).toFixed(1),
            (prediction.awayWin * 100).toFixed(1)
          ],
          backgroundColor: ['#36A2EB', '#FFCE56', '#FF6384'],
          borderWidth: 1,
        },
      ],
    };
    
    return (
      <Pie 
        data={data}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: 'bottom',
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return `${context.label}: ${context.raw}%`;
                }
              }
            }
          }
        }}
      />
    );
  };

  // Get the most likely outcome
  const getMostLikelyOutcome = () => {
    if (!prediction) return null;
    
    const outcomes = [
      { name: 'Home Win', probability: prediction.homeWin },
      { name: 'Draw', probability: prediction.draw },
      { name: 'Away Win', probability: prediction.awayWin }
    ];
    
    const mostLikely = outcomes.reduce((max, outcome) => 
      outcome.probability > max.probability ? outcome : max, outcomes[0]);
    
    return mostLikely;
  };

  const selectedHomeTeam = teams.find(team => team.id === parseInt(homeTeam));
  const selectedAwayTeam = teams.find(team => team.id === parseInt(awayTeam));
  const mostLikelyOutcome = getMostLikelyOutcome();

  return (
    <Container>
      <h1 className="mb-4">Match Prediction</h1>
      
      <Row>
        <Col md={5}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>Select Teams</Card.Title>
              {error && <Alert variant="danger">{error}</Alert>}
              
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>Home Team</Form.Label>
                  <Form.Select
                    value={homeTeam}
                    onChange={(e) => setHomeTeam(e.target.value)}
                    required
                  >
                    <option value="">Select Home Team</option>
                    {teams.map(team => (
                      <option key={`home-${team.id}`} value={team.id}>
                        {team.name}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Away Team</Form.Label>
                  <Form.Select
                    value={awayTeam}
                    onChange={(e) => setAwayTeam(e.target.value)}
                    required
                  >
                    <option value="">Select Away Team</option>
                    {teams.map(team => (
                      <option key={`away-${team.id}`} value={team.id}>
                        {team.name}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Prediction Model</Form.Label>
                  <Form.Select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                  >
                    <option value="team-based">Team-Based Model</option>
                    <option value="xgboost">XGBoost Model</option>
                  </Form.Select>
                </Form.Group>
                
                <Button variant="primary" type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <Spinner as="span" animation="border" size="sm" className="me-2" />
                      Predicting...
                    </>
                  ) : 'Predict Match'}
                </Button>
              </Form>
            </Card.Body>
          </Card>
          
          <Card>
            <Card.Body>
              <Card.Title>How It Works</Card.Title>
              <Card.Text>
                Our prediction models analyze team attributes, recent form, and historical match data
                to predict the most likely outcome of a match between the selected teams.
              </Card.Text>
              <Card.Text>
                <strong>Team-Based Model:</strong> Neural network that focuses on team attributes and form.
              </Card.Text>
              <Card.Text>
                <strong>XGBoost Model:</strong> Machine learning model that considers detailed match statistics.
              </Card.Text>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={7}>
          {prediction ? (
            <Card>
              <Card.Body>
                <Card.Title>
                  Match Prediction: {selectedHomeTeam?.name} vs {selectedAwayTeam?.name}
                </Card.Title>
                
                <div className="text-center mb-4">
                  <div className="d-block my-4" style={{ maxWidth: '350px', margin: '0 auto' }}>
                    {getPredictionChart()}
                  </div>
                </div>
                
                <Card.Text className="text-center">
                  <strong>Most Likely Outcome:</strong> {mostLikelyOutcome?.name} ({(mostLikelyOutcome?.probability * 100).toFixed(1)}%)
                </Card.Text>
                
                <Row className="text-center">
                  <Col>
                    <div className="p-3 bg-light rounded">
                      <h5>{selectedHomeTeam?.name}</h5>
                      <h3>{(prediction.homeWin * 100).toFixed(1)}%</h3>
                      <p>Win Probability</p>
                    </div>
                  </Col>
                  <Col>
                    <div className="p-3 bg-light rounded">
                      <h5>Draw</h5>
                      <h3>{(prediction.draw * 100).toFixed(1)}%</h3>
                      <p>Draw Probability</p>
                    </div>
                  </Col>
                  <Col>
                    <div className="p-3 bg-light rounded">
                      <h5>{selectedAwayTeam?.name}</h5>
                      <h3>{(prediction.awayWin * 100).toFixed(1)}%</h3>
                      <p>Win Probability</p>
                    </div>
                  </Col>
                </Row>
                
                <hr />
                
                <div className="text-muted">
                  <small>
                    Note: This prediction is based on historical data and model training.
                    Actual match outcomes are influenced by many factors and may differ.
                  </small>
                </div>
              </Card.Body>
            </Card>
          ) : (
            <div className="text-center p-5 bg-light rounded">
              <h4>Select teams and click "Predict Match" to see the prediction</h4>
              <p className="text-muted">
                The prediction will show win probabilities for both teams and the chance of a draw.
              </p>
            </div>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default PredictionPage; 