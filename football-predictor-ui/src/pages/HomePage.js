import React from 'react';
import { Container, Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <Container>
      <Row className="mb-4">
        <Col>
          <div className="jumbotron text-center bg-light p-5 rounded">
            <h1 className="display-4">Football Match Prediction</h1>
            <p className="lead">
              Using machine learning to predict football match outcomes with advanced neural networks and XGBoost models.
            </p>
            <p>
              <Button as={Link} to="/predict" variant="primary" size="lg" className="me-2">
                Try Predictions
              </Button>
              <Button as={Link} to="/models" variant="secondary" size="lg">
                Learn About Models
              </Button>
            </p>
          </div>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={4} className="mb-3">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Team-Based Model</Card.Title>
              <Card.Text>
                Our neural network model analyzes team attributes and recent form to predict match outcomes.
              </Card.Text>
              <Button as={Link} to="/models" variant="outline-primary">Model Details</Button>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4} className="mb-3">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Match Predictions</Card.Title>
              <Card.Text>
                Select any two teams and get instant predictions based on our advanced models.
              </Card.Text>
              <Button as={Link} to="/predict" variant="outline-primary">Make Predictions</Button>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4} className="mb-3">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Team Analysis</Card.Title>
              <Card.Text>
                Explore team tactical attributes and form analysis for all major football teams.
              </Card.Text>
              <Button as={Link} to="/teams" variant="outline-primary">Analyze Teams</Button>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>About the Project</Card.Title>
              <Card.Text>
                This project is part of a thesis at ELTE University, focusing on applying machine learning to football match prediction.
                The models have been trained on historical match data and can predict with approximately 59% accuracy, which is better than random guessing (33%).
              </Card.Text>
              <Card.Text>
                The system uses multiple prediction models:
              </Card.Text>
              <ul>
                <li>A neural network approach that analyzes team attributes and recent form</li>
                <li>XGBoost models that consider detailed match statistics</li>
              </ul>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default HomePage; 