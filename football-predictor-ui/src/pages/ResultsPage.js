import React, { useState } from 'react';
import { Container, Row, Col, Card, Table, Form } from 'react-bootstrap';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const ResultsPage = () => {
  const [selectedModel, setSelectedModel] = useState('team-based');
  const [selectedLeague, setSelectedLeague] = useState('all');
  
  // Sample data - in a real app, this would come from an API
  const modelAccuracyByLeague = {
    'team-based': {
      'all': 59.0,
      'premier-league': 62.5,
      'la-liga': 58.7,
      'bundesliga': 61.2,
      'serie-a': 57.8,
      'ligue-1': 56.4
    },
    'xgboost': {
      'all': 62.0,
      'premier-league': 64.1,
      'la-liga': 61.5,
      'bundesliga': 63.7,
      'serie-a': 60.2,
      'ligue-1': 59.8
    }
  };
  
  const confusionMatrix = {
    'team-based': {
      'away_win': [204, 44, 50],
      'draw': [69, 135, 38],
      'home_win': [127, 58, 275]
    },
    'xgboost': {
      'away_win': [215, 38, 45],
      'draw': [62, 142, 38],
      'home_win': [110, 52, 298]
    }
  };
  
  const predictionExamples = [
    {
      id: 1,
      homeTeam: 'Manchester City',
      awayTeam: 'Liverpool',
      actualResult: '2-1',
      outcome: 'Home Win',
      teamBasedPrediction: { home: 0.65, draw: 0.20, away: 0.15 },
      xgboostPrediction: { home: 0.58, draw: 0.25, away: 0.17 },
      correct: true
    },
    {
      id: 2,
      homeTeam: 'Barcelona',
      awayTeam: 'Real Madrid',
      actualResult: '1-3',
      outcome: 'Away Win',
      teamBasedPrediction: { home: 0.45, draw: 0.25, away: 0.30 },
      xgboostPrediction: { home: 0.40, draw: 0.23, away: 0.37 },
      correct: false
    },
    {
      id: 3,
      homeTeam: 'Bayern Munich',
      awayTeam: 'Borussia Dortmund',
      actualResult: '2-2',
      outcome: 'Draw',
      teamBasedPrediction: { home: 0.55, draw: 0.25, away: 0.20 },
      xgboostPrediction: { home: 0.52, draw: 0.28, away: 0.20 },
      correct: false
    },
    {
      id: 4,
      homeTeam: 'Juventus',
      awayTeam: 'AC Milan',
      actualResult: '1-0',
      outcome: 'Home Win',
      teamBasedPrediction: { home: 0.48, draw: 0.32, away: 0.20 },
      xgboostPrediction: { home: 0.51, draw: 0.29, away: 0.20 },
      correct: true
    },
    {
      id: 5,
      homeTeam: 'Chelsea',
      awayTeam: 'Arsenal',
      actualResult: '0-0',
      outcome: 'Draw',
      teamBasedPrediction: { home: 0.42, draw: 0.33, away: 0.25 },
      xgboostPrediction: { home: 0.38, draw: 0.35, away: 0.27 },
      correct: true
    }
  ];
  
  // Helper function to get confusion matrix chart
  const getConfusionMatrixChart = () => {
    const matrix = confusionMatrix[selectedModel];
    
    // Calculate class-wise accuracy (precision)
    const awayTotal = matrix['away_win'][0] + matrix['draw'][0] + matrix['home_win'][0];
    const drawTotal = matrix['away_win'][1] + matrix['draw'][1] + matrix['home_win'][1];
    const homeTotal = matrix['away_win'][2] + matrix['draw'][2] + matrix['home_win'][2];
    
    const awayPrecision = (matrix['away_win'][0] / awayTotal) * 100;
    const drawPrecision = (matrix['draw'][1] / drawTotal) * 100;
    const homePrecision = (matrix['home_win'][2] / homeTotal) * 100;
    
    // Calculate class-wise recall
    const awayRecall = (matrix['away_win'][0] / (matrix['away_win'][0] + matrix['away_win'][1] + matrix['away_win'][2])) * 100;
    const drawRecall = (matrix['draw'][1] / (matrix['draw'][0] + matrix['draw'][1] + matrix['draw'][2])) * 100;
    const homeRecall = (matrix['home_win'][2] / (matrix['home_win'][0] + matrix['home_win'][1] + matrix['home_win'][2])) * 100;
    
    // Bar chart for precision and recall
    const barData = {
      labels: ['Away Win', 'Draw', 'Home Win'],
      datasets: [
        {
          label: 'Precision (%)',
          data: [awayPrecision.toFixed(1), drawPrecision.toFixed(1), homePrecision.toFixed(1)],
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
        },
        {
          label: 'Recall (%)',
          data: [awayRecall.toFixed(1), drawRecall.toFixed(1), homeRecall.toFixed(1)],
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1,
        }
      ],
    };
    
    return (
      <Bar 
        data={barData}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: 'top',
            },
            title: {
              display: true,
              text: 'Precision and Recall by Outcome',
            },
          },
        }}
      />
    );
  };
  
  // Helper function to get league accuracy chart
  const getLeagueAccuracyChart = () => {
    const modelData = modelAccuracyByLeague[selectedModel];
    
    const data = {
      labels: ['All Leagues', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
      datasets: [
        {
          data: [
            modelData['all'],
            modelData['premier-league'],
            modelData['la-liga'],
            modelData['bundesliga'],
            modelData['serie-a'],
            modelData['ligue-1']
          ],
          backgroundColor: [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)',
            'rgba(255, 159, 64, 0.7)',
          ],
          borderWidth: 1,
        },
      ],
    };
    
    return (
      <Doughnut 
        data={data}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: 'right',
            },
            title: {
              display: true,
              text: 'Accuracy by League (%)',
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return `${context.label}: ${context.raw}%`;
                }
              }
            }
          },
        }}
      />
    );
  };

  return (
    <Container>
      <h1 className="mb-4">Model Performance Results</h1>
      
      <Row className="mb-4">
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Select Model</Form.Label>
            <Form.Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="team-based">Team-Based Model</option>
              <option value="xgboost">XGBoost Model</option>
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Select League</Form.Label>
            <Form.Select
              value={selectedLeague}
              onChange={(e) => setSelectedLeague(e.target.value)}
            >
              <option value="all">All Leagues</option>
              <option value="premier-league">Premier League</option>
              <option value="la-liga">La Liga</option>
              <option value="bundesliga">Bundesliga</option>
              <option value="serie-a">Serie A</option>
              <option value="ligue-1">Ligue 1</option>
            </Form.Select>
          </Form.Group>
        </Col>
      </Row>
      
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>Model Accuracy</Card.Title>
              <div className="text-center">
                <h2>{modelAccuracyByLeague[selectedModel][selectedLeague]}%</h2>
                <p className="text-muted">
                  Overall accuracy for {selectedModel === 'team-based' ? 'Team-Based' : 'XGBoost'} model
                  {selectedLeague !== 'all' ? ` in ${selectedLeague.replace('-', ' ')}` : ''}
                </p>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row className="mb-4">
        <Col md={7}>
          <Card>
            <Card.Body>
              <Card.Title>Prediction Performance by Outcome</Card.Title>
              <div className="mt-3">
                {getConfusionMatrixChart()}
              </div>
            </Card.Body>
          </Card>
        </Col>
        <Col md={5}>
          <Card>
            <Card.Body>
              <Card.Title>Performance by League</Card.Title>
              <div className="mt-3" style={{ height: '300px' }}>
                {getLeagueAccuracyChart()}
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>Recent Prediction Examples</Card.Title>
              <Table striped bordered hover responsive>
                <thead>
                  <tr>
                    <th>Match</th>
                    <th>Actual Result</th>
                    <th>Team-Based Prediction</th>
                    <th>XGBoost Prediction</th>
                    <th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionExamples.map(example => (
                    <tr key={example.id}>
                      <td>{example.homeTeam} vs {example.awayTeam}</td>
                      <td>{example.actualResult}</td>
                      <td>
                        Home: {(example.teamBasedPrediction.home * 100).toFixed(0)}%<br />
                        Draw: {(example.teamBasedPrediction.draw * 100).toFixed(0)}%<br />
                        Away: {(example.teamBasedPrediction.away * 100).toFixed(0)}%
                      </td>
                      <td>
                        Home: {(example.xgboostPrediction.home * 100).toFixed(0)}%<br />
                        Draw: {(example.xgboostPrediction.draw * 100).toFixed(0)}%<br />
                        Away: {(example.xgboostPrediction.away * 100).toFixed(0)}%
                      </td>
                      <td>
                        <span className={example.correct ? 'text-success' : 'text-danger'}>
                          {example.outcome}
                          {example.correct ? ' ✓' : ' ✗'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default ResultsPage; 