import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Table, Tabs, Tab } from 'react-bootstrap';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const TeamsPage = () => {
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [teamData, setTeamData] = useState(null);
  
  // In a real app, this would come from an API call
  useEffect(() => {
    // Simulated teams data
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

  // When a team is selected, fetch its data
  useEffect(() => {
    if (!selectedTeam) return;
    
    // Simulated team data - in a real app, this would be fetched from the backend
    const generateTeamData = () => {
      return {
        id: parseInt(selectedTeam),
        name: teams.find(t => t.id === parseInt(selectedTeam))?.name,
        attributes: {
          buildupSpeed: Math.floor(Math.random() * 40) + 60, // 60-99
          buildupPassing: Math.floor(Math.random() * 40) + 60,
          chanceCreation: Math.floor(Math.random() * 40) + 60,
          defencePressure: Math.floor(Math.random() * 40) + 60,
          defenceAggression: Math.floor(Math.random() * 40) + 60,
          defenceWidth: Math.floor(Math.random() * 40) + 60
        },
        recentForm: {
          wins: Math.floor(Math.random() * 6), // 0-5
          draws: Math.floor(Math.random() * 3), // 0-2
          losses: Math.floor(Math.random() * 3), // 0-2
          goalsScored: Math.floor(Math.random() * 16) + 5, // 5-20
          goalsConceded: Math.floor(Math.random() * 10) + 2, // 2-11
          cleanSheets: Math.floor(Math.random() * 4) // 0-3
        },
        lastFive: [
          { opponent: 'Team A', result: 'W', score: '2-0' },
          { opponent: 'Team B', result: 'D', score: '1-1' },
          { opponent: 'Team C', result: 'W', score: '3-1' },
          { opponent: 'Team D', result: 'L', score: '0-2' },
          { opponent: 'Team E', result: 'W', score: '1-0' }
        ]
      };
    };
    
    setTeamData(generateTeamData());
  }, [selectedTeam, teams]);

  // Prepare radar chart for team attributes
  const getAttributesChart = () => {
    if (!teamData) return null;
    
    const data = {
      labels: [
        'Buildup Speed', 
        'Buildup Passing', 
        'Chance Creation', 
        'Defence Pressure', 
        'Defence Aggression', 
        'Defence Width'
      ],
      datasets: [
        {
          label: teamData.name,
          data: [
            teamData.attributes.buildupSpeed,
            teamData.attributes.buildupPassing,
            teamData.attributes.chanceCreation,
            teamData.attributes.defencePressure,
            teamData.attributes.defenceAggression,
            teamData.attributes.defenceWidth
          ],
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgb(54, 162, 235)',
          pointBackgroundColor: 'rgb(54, 162, 235)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(54, 162, 235)'
        }
      ]
    };
    
    return (
      <Radar 
        data={data}
        options={{
          scales: {
            r: {
              min: 0,
              max: 100,
              ticks: {
                stepSize: 20
              }
            }
          }
        }}
      />
    );
  };

  return (
    <Container>
      <h1 className="mb-4">Team Analysis</h1>
      
      <Row className="mb-4">
        <Col md={4}>
          <Card>
            <Card.Body>
              <Card.Title>Select a Team</Card.Title>
              <Form.Select
                value={selectedTeam}
                onChange={(e) => setSelectedTeam(e.target.value)}
              >
                <option value="">Select Team</option>
                {teams.map(team => (
                  <option key={team.id} value={team.id}>
                    {team.name}
                  </option>
                ))}
              </Form.Select>
            </Card.Body>
          </Card>
        </Col>
        {teamData && (
          <Col md={8}>
            <Card className="text-center">
              <Card.Body>
                <Card.Title className="display-6">{teamData.name}</Card.Title>
                <div className="d-flex justify-content-center mt-3 mb-2">
                  <div className="px-3 border-end">
                    <h5>Wins</h5>
                    <h3>{teamData.recentForm.wins}</h3>
                  </div>
                  <div className="px-3 border-end">
                    <h5>Draws</h5>
                    <h3>{teamData.recentForm.draws}</h3>
                  </div>
                  <div className="px-3">
                    <h5>Losses</h5>
                    <h3>{teamData.recentForm.losses}</h3>
                  </div>
                </div>
              </Card.Body>
            </Card>
          </Col>
        )}
      </Row>
      
      {teamData && (
        <Tabs defaultActiveKey="attributes" className="mb-3">
          <Tab eventKey="attributes" title="Team Attributes">
            <Row>
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <Card.Title>Tactical Attributes</Card.Title>
                    <div className="mt-3">
                      {getAttributesChart()}
                    </div>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <Card.Title>Attribute Details</Card.Title>
                    <Table striped>
                      <thead>
                        <tr>
                          <th>Attribute</th>
                          <th>Rating</th>
                          <th>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Buildup Speed</td>
                          <td>{teamData.attributes.buildupSpeed}</td>
                          <td>Speed at which the team builds up play from defense to attack</td>
                        </tr>
                        <tr>
                          <td>Buildup Passing</td>
                          <td>{teamData.attributes.buildupPassing}</td>
                          <td>Passing style during buildup phase (direct vs. possession)</td>
                        </tr>
                        <tr>
                          <td>Chance Creation</td>
                          <td>{teamData.attributes.chanceCreation}</td>
                          <td>Team's effectiveness at creating scoring opportunities</td>
                        </tr>
                        <tr>
                          <td>Defence Pressure</td>
                          <td>{teamData.attributes.defencePressure}</td>
                          <td>Level of pressure applied to opponents when defending</td>
                        </tr>
                        <tr>
                          <td>Defence Aggression</td>
                          <td>{teamData.attributes.defenceAggression}</td>
                          <td>Aggression level in defensive actions and tackles</td>
                        </tr>
                        <tr>
                          <td>Defence Width</td>
                          <td>{teamData.attributes.defenceWidth}</td>
                          <td>How wide or narrow the team defends</td>
                        </tr>
                      </tbody>
                    </Table>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          </Tab>
          
          <Tab eventKey="form" title="Recent Form">
            <Row>
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <Card.Title>Form Statistics</Card.Title>
                    <Table striped>
                      <thead>
                        <tr>
                          <th>Statistic</th>
                          <th>Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Goals Scored</td>
                          <td>{teamData.recentForm.goalsScored}</td>
                        </tr>
                        <tr>
                          <td>Goals Conceded</td>
                          <td>{teamData.recentForm.goalsConceded}</td>
                        </tr>
                        <tr>
                          <td>Clean Sheets</td>
                          <td>{teamData.recentForm.cleanSheets}</td>
                        </tr>
                        <tr>
                          <td>Win Rate</td>
                          <td>
                            {Math.round(teamData.recentForm.wins / 
                              (teamData.recentForm.wins + teamData.recentForm.draws + teamData.recentForm.losses) * 100)}%
                          </td>
                        </tr>
                        <tr>
                          <td>Goal Difference</td>
                          <td>{teamData.recentForm.goalsScored - teamData.recentForm.goalsConceded}</td>
                        </tr>
                        <tr>
                          <td>Points per Game</td>
                          <td>
                            {((teamData.recentForm.wins * 3 + teamData.recentForm.draws) / 
                              (teamData.recentForm.wins + teamData.recentForm.draws + teamData.recentForm.losses)).toFixed(2)}
                          </td>
                        </tr>
                      </tbody>
                    </Table>
                  </Card.Body>
                </Card>
              </Col>
              
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <Card.Title>Last 5 Matches</Card.Title>
                    <Table striped>
                      <thead>
                        <tr>
                          <th>Opponent</th>
                          <th>Result</th>
                          <th>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {teamData.lastFive.map((match, index) => (
                          <tr key={index}>
                            <td>{match.opponent}</td>
                            <td>
                              <span className={
                                match.result === 'W' ? 'text-success' : 
                                match.result === 'D' ? 'text-warning' : 
                                'text-danger'
                              }>
                                {match.result === 'W' ? 'Win' : match.result === 'D' ? 'Draw' : 'Loss'}
                              </span>
                            </td>
                            <td>{match.score}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          </Tab>
        </Tabs>
      )}
    </Container>
  );
};

export default TeamsPage; 