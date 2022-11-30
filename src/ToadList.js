import Container from "react-bootstrap/Container";

const Toad = (props) => {
  return <Container className='center'>
    <h3>{props.label}</h3>
    <p>{props.description}</p>
    <img alt='toad' className='toadimg' src={props.image}/>
    <hr className='coral-black'/>
  </Container>
}

const ToadList = (props) => {
  let len = props.images.length
  let toads = []

  for (let i = 0; i < len; i++) {
    toads.push(<Toad key={i} label={props.labels[i]} image={props.images[i]} description={props.descriptions[i]}/>)
  }

  return <Container>{toads}</Container>
}

export default ToadList