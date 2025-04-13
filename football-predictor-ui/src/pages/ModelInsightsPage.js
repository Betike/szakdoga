import React, { useState } from 'react';
import { Container, Row, Col, Card, Tabs, Tab, Table, Alert } from 'react-bootstrap';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelInsightsPage = () => {
  const [activeTab, setActiveTab] = useState('team-based');

  // Team based model feature importance (from our analysis)
  const teamBasedFeatures = [
    { feature: 'home_defence_aggression', importance: 13.63 },
    { feature: 'home_chance_creation', importance: 11.08 },
    { feature: 'away_buildup_passing', importance: 10.60 },
    { feature: 'away_defence_aggression', importance: 10.24 },
    { feature: 'away_chance_creation', importance: 9.45 },
    { feature: 'home_defence_width', importance: 7.93 },
    { feature: 'away_buildup_speed', importance: 7.75 },
    { feature: 'away_defence_pressure', importance: 7.75 },
    { feature: 'home_buildup_passing', importance: 7.27 },
    { feature: 'home_defence_pressure', importance: 5.09 }
  ];

  // Category importance
  const categoryImportance = [
    { category: 'Away Team Attributes', percentage: 50.39 },
    { category: 'Home Team Attributes', percentage: 49.61 },
    { category: 'Away Team Goal Stats', percentage: 0.00 },
    { category: 'Away Team Match Results', percentage: 0.00 },
    { category: 'Home Team Goal Stats', percentage: 0.00 },
    { category: 'Home Team Match Results', percentage: 0.00 }
  ];

  // XGBoost model sample feature importance (fictional data)
  const xgboostFeatures = [
    { feature: 'home_team_rating', importance: 18.5 },
    { feature: 'away_team_rating', importance: 17.2 },
    { feature: 'home_win_streak', importance: 14.3 },
    { feature: 'home_goals_scored_avg', importance: 10.8 },
    { feature: 'away_goals_conceded_avg', importance: 9.5 },
    { feature: 'home_team_form', importance: 8.7 },
    { feature: 'away_team_form', importance: 7.9 },
    { feature: 'away_goals_scored_avg', importance: 6.1 },
    { feature: 'home_goals_conceded_avg', importance: 4.6 },
    { feature: 'derby_match', importance: 2.4 }
  ];

  // Model performance metrics (fictional data - replace with actual metrics)
  const modelPerformance = {
    'team-based': {
      accuracy: 59.0,
      homeWinPrecision: 68.0,
      drawPrecision: 58.0,
      awayWinPrecision: 51.0,
      homeWinRecall: 53.0,
      drawRecall: 56.0,
      awayWinRecall: 69.0
    },
    'xgboost': {
      accuracy: 62.0,
      homeWinPrecision: 70.0,
      drawPrecision: 55.0,
      awayWinPrecision: 56.0,
      homeWinRecall: 60.0,
      drawRecall: 50.0,
      awayWinRecall: 65.0
    }
  };

  // Prepare bar chart data for feature importance
  const getFeatureImportanceChart = (features) => {
    const data = {
      labels: features.map(f => f.feature),
      datasets: [
        {
          label: 'Importance (%)',
          data: features.map(f => f.importance),
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
        },
      ],
    };
    
    return (
      <Bar 
        data={data}
        options={{
          indexAxis: 'y',
          responsive: true,
          plugins: {
            legend: {
              position: 'top',
            },
            title: {
              display: true,
              text: 'Feature Importance (%)',
            },
          },
        }}
      />
    );
  };

  // Prepare bar chart for category importance
  const getCategoryImportanceChart = () => {
    const data = {
      labels: categoryImportance.map(c => c.category),
      datasets: [
        {
          label: 'Importance (%)',
          data: categoryImportance.map(c => c.percentage),
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
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
      <Bar 
        data={data}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: 'top',
            },
            title: {
              display: true,
              text: 'Feature Category Importance (%)',
            },
          },
        }}
      />
    );
  };

  return (
    <Container>
      <h1 className="mb-4">Model Insights</h1>
      
      <Alert variant="info">
        This page shows insights into how our prediction models work, focusing on the most important factors
        that influence match predictions.
      </Alert>
      
      <Tabs
        activeKey={activeTab}
        onSelect={(key) => setActiveTab(key)}
        className="mb-4"
      >
        <Tab eventKey="team-based" title="Team-Based Model">
          <Row className="mb-4">
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>Team-Based Neural Network Model</Card.Title>
                  <Card.Text>
                    This neural network model analyzes team attributes and recent form to predict match outcomes.
                    It uses separate encoders for team attributes and form data, then combines these features
                    to make predictions.
                  </Card.Text>
                  <Card.Text>
                    <strong>Key finding:</strong> Team tactical attributes dominate the prediction importance, with
                    defensive aggression and chance creation being the most influential factors.
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col md={7}>
              <Card>
                <Card.Body>
                  <Card.Title>Top 10 Most Important Features</Card.Title>
                  <div className="mt-3">
                    {getFeatureImportanceChart(teamBasedFeatures)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
            <Col md={5}>
              <Card>
                <Card.Body>
                  <Card.Title>Feature Category Importance</Card.Title>
                  <div className="mt-3">
                    {getCategoryImportanceChart()}
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <Row>
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>Model Performance</Card.Title>
                  <Table striped bordered hover className="mt-3">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Accuracy</td>
                        <td>{modelPerformance['team-based'].accuracy}%</td>
                        <td>Overall prediction accuracy across all match outcomes</td>
                      </tr>
                      <tr>
                        <td>Home Win Precision</td>
                        <td>{modelPerformance['team-based'].homeWinPrecision}%</td>
                        <td>When model predicts a home win, it's correct this percentage of time</td>
                      </tr>
                      <tr>
                        <td>Draw Precision</td>
                        <td>{modelPerformance['team-based'].drawPrecision}%</td>
                        <td>When model predicts a draw, it's correct this percentage of time</td>
                      </tr>
                      <tr>
                        <td>Away Win Precision</td>
                        <td>{modelPerformance['team-based'].awayWinPrecision}%</td>
                        <td>When model predicts an away win, it's correct this percentage of time</td>
                      </tr>
                    </tbody>
                  </Table>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Tab>
        
        <Tab eventKey="xgboost" title="XGBoost Model">
          <Row className="mb-4">
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>XGBoost Model</Card.Title>
                  <Card.Text>
                    Our XGBoost model uses detailed match statistics and team-related features to predict
                    match outcomes. XGBoost is a powerful gradient boosting framework that works well with
                    tabular data.
                  </Card.Text>
                  <Card.Text>
                    <strong>Key finding:</strong> Team ratings and win streaks are the most influential factors
                    in XGBoost predictions, followed by goals scored and conceded averages.
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>Top 10 Most Important Features</Card.Title>
                  <div className="mt-3">
                    {getFeatureImportanceChart(xgboostFeatures)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <Row>
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>Model Performance</Card.Title>
                  <Table striped bordered hover className="mt-3">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Accuracy</td>
                        <td>{modelPerformance['xgboost'].accuracy}%</td>
                        <td>Overall prediction accuracy across all match outcomes</td>
                      </tr>
                      <tr>
                        <td>Home Win Precision</td>
                        <td>{modelPerformance['xgboost'].homeWinPrecision}%</td>
                        <td>When model predicts a home win, it's correct this percentage of time</td>
                      </tr>
                      <tr>
                        <td>Draw Precision</td>
                        <td>{modelPerformance['xgboost'].drawPrecision}%</td>
                        <td>When model predicts a draw, it's correct this percentage of time</td>
                      </tr>
                      <tr>
                        <td>Away Win Precision</td>
                        <td>{modelPerformance['xgboost'].awayWinPrecision}%</td>
                        <td>When model predicts an away win, it's correct this percentage of time</td>
                      </tr>
                    </tbody>
                  </Table>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Tab>
      </Tabs>
    </Container>
  );
};

export default ModelInsightsPage; 