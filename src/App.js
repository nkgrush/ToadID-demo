import 'bootstrap/dist/css/bootstrap.min.css';

import logo from './frog.png';
import example_crop from './example-crop.png'
import example_toad from './example-toad.jpeg'
import './App.css';
import Markdown from 'react-markdown'
import Image from 'react-bootstrap/Image'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Button from 'react-bootstrap/Button'
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import {PureComponent} from "react";
import ToadList from './ToadList'

let speech = "# Инструкция\n" +
  "1. Загрузите картинку жабы\n" +
  "2. Выделите квадратную область ниже паротид\n" +
  "3. После нажатия на кнопку, через пару секунд cайт покажет **5 самых похожих жаб** 🌈 из базы с их именами и текстовыми описаниями\n."

class App extends PureComponent {
  state = {
    src: null,
    crop: {
      unit: '%',
      width: 29,
      aspect: 1
    }
  };

  componentDidMount() {
    document.title = 'ToadID'
  }

  handleClick = async () => {
    this.setState({isLoadingToads: true})
    const url = this.state.croppedImageUrl
    let file = await fetch(url)
      .then(r => r.blob())
      .then(blobFile => new File(
        [blobFile], "crop", { type: "image/png" }
      ))

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload_image', {method: 'POST', body: formData})
      .then(r => r.json())
      .then(r => {
        this.setState({isLoadingToads: false, images: r.images, labels: r.labels, descriptions: r.descriptions})
      })
  }

  onSelectFile = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const reader = new FileReader();
      reader.addEventListener('load', () =>
        this.setState({src: reader.result})
      );
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  onImageLoaded = (image) => {
    this.imageRef = image;
  };

  onCropComplete = (crop) => {
    this.makeClientCrop(crop);
  };

  onCropChange = (crop, percentCrop) => {
    // You could also use percentCrop:
    // this.setState({ crop: percentCrop });
    this.setState({crop});
  };

  async makeClientCrop(crop) {
    if (this.imageRef && crop.width && crop.height) {
      const croppedImageUrl = await this.getCroppedImg(
        this.imageRef,
        crop,
        'crop.png'
      );
      this.setState({ croppedImageUrl });
    }
  }

  getCroppedImg(image, crop, fileName) {
    const canvas = document.createElement('canvas');
    const pixelRatio = window.devicePixelRatio;
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    const ctx = canvas.getContext('2d');

    canvas.width = crop.width * pixelRatio * scaleX;
    canvas.height = crop.height * pixelRatio * scaleY;

    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.imageSmoothingQuality = 'high';

    ctx.drawImage(
      image,
      crop.x * scaleX,
      crop.y * scaleY,
      crop.width * scaleX,
      crop.height * scaleY,
      0,
      0,
      crop.width * scaleX,
      crop.height * scaleY
    );

    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            //reject(new Error('Canvas is empty'));
            console.error('Canvas is empty');
            return;
          }
          blob.name = fileName;
          window.URL.revokeObjectURL(this.fileUrl);
          this.fileUrl = window.URL.createObjectURL(blob);
          resolve(this.fileUrl);
        },
        'image/png',
        1
      );
    });
  }


  render()
  {
    const { crop, croppedImageUrl, src } = this.state;
    return (
      <div className="App">
        <header className="App-header">
          <div className='container title-fit'>
            <div className={'title'}>🐸Toad ID</div>
          </div>
        </header>
        <div className='container body' style={{width: '80%'}}>
          <br/>
          <div id='cropper' className='coral-black'>
            <input type="file" accept="image/*" onChange={this.onSelectFile} />
          </div>
          <hr/>
          <Row md={2} xs={1}>
            <Col>
              {src && (
                <ReactCrop
                  src={src}
                  crop={crop}
                  ruleOfThirds
                  onImageLoaded={this.onImageLoaded}
                  onComplete={this.onCropComplete}
                  onChange={this.onCropChange}
                />
              )}
            </Col>
            <Col>
              {croppedImageUrl && (
                <img alt="Crop" style={{ maxWidth: '100%' }} src={croppedImageUrl} />
              )}
            </Col>
          </Row>

          {src &&
          <div>
            <div className='button-holder'>
              <Button onClick={this.handleClick} disabled={this.state.isLoadingToads} variant="primary">{this.state.isLoadingToads ? 'Waiting...' : 'Upload!'}</Button>{' '}
            </div>
            <hr className='coral-black'/>
          </div>
          }

          {this.state.images &&
          <ToadList images={this.state.images} labels={this.state.labels} descriptions={this.state.descriptions}/>
          }
          <Markdown children={speech}/>
          <Container style={{width: '70%'}}>
            <Row md={2} xs={1}>
              <Col><Image src={example_toad} fluid/></Col>
              <Col><p className='boxtext'>База данных содержит фотографии жаб со спины. Они могут быть использованы для ручного сравнения, а также для того чтобы посмотреть на животное целиком.</p></Col>
            </Row>
            <Row md={2} xs={1}>
              <Col><p className='boxtext'>Нейросеть была обучена на квадратных фрагментах изображения, начинающихся сразу за паротидами. После загрузки фотографии, сайт предлагает обрезать её до нужного размера.</p></Col>
              <Col><Image src={example_crop} fluid/></Col>
            </Row>
          </Container>

        </div>
        <div className={'container kek'}> </div>
        <footer>
          <p className='container'>Грушецкий Николай, 2021</p>
        </footer>
      </div>
    );
  }
}

export default App;
