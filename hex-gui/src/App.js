import logo from './logo.svg';
import gameExplanation from './GameExplanation.png';

import './App.css';
import './Hexagon.css';
import range from './arrayTools';

function Hexagon(props) {
  // const basePoints = [[0,30],[26,15],[26,-15],[0,-30],[-26,-15],[-26,15]];
  const basePoints = [[26, 60], [52, 45], [52, 15], [26, 0], [0, 15], [0, 45]];
  const totalWidth = 52;
  const doubleRowHeight = 60 + 30;
  const secondRowXOffset = 26;
  const secondRowYOffset = 45;
  const pointsWithPosition = basePoints.map(c => [
    c[0] + totalWidth * props.gridPositionX + secondRowXOffset * (props.gridPositionY % 2 == 1),
    c[1] + doubleRowHeight * props.gridPositionY / 2 + secondRowYOffset * (props.gridPositionY % 2 > 1)
  ]);
  const points = pointsWithPosition.map(c => [c[0] * props.gridSize, c[1] * props.gridSize]);
  const pointsString = points.map(c => c[0] + "," + c[1]).join(" ");
  return <polygon className="Hexagon" points={pointsString} fill={props.fill}></polygon>
}


function App() {
  const rows = [
    range(5, 6),
    range(4, 6),
    range(4, 7),
    range(3, 7),
    range(3, 8),
    range(2, 8),
    range(2, 9),
    range(1, 9),
    range(1, 10),
    range(0, 10),
    range(0, 11),
    range(0, 10),
    range(1, 10),
    range(1, 9),
    range(2, 9),
    range(2, 8),
    range(3, 8),
    range(3, 7),
    range(4, 7),
    range(4, 6),
    range(5, 6)
  ];

  return <div className="App">
    <h1>Play</h1>
    <p>Rules:</p>
    <img src={gameExplanation} />
    <p>Have fun:</p>
    <svg viewBox="0 0 1500 1500" xmlns="http://www.w3.org/2000/svg">
      {rows.map((row, rowIndex) => 
          row.map(colIndex => 
            <Hexagon key={rowIndex * 1000 + colIndex} gridPositionX={colIndex} gridPositionY={rowIndex} gridSize="1" fill="white"></Hexagon>
          )
      )};
      {/* <Hexagon gridPositionX="0" gridPositionY="0" gridSize="2" fill="blue"></Hexagon>
      <Hexagon gridPositionX="1" gridPositionY="0" gridSize="2" fill="red"></Hexagon>
      <Hexagon gridPositionX="0" gridPositionY="1" gridSize="2" fill="green"></Hexagon> */}

    </svg>
  </div>;
}

export default App;
