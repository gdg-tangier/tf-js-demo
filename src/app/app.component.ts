import { Component } from '@angular/core';

import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app';
  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit () {
    this.train ()
  }

  async train () {
    this.linearModel = tf.sequential();

    this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const x = tf.tensor1d([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]);
    const y = tf.tensor1d([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]);

    await this.linearModel.fit(x, y);


    console.log("model Trained")
  }

  predict (val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1,1])) as any;

    this.prediction = Array.from(output.dataSync())[0]
    console.log(this.prediction)
  }
}
