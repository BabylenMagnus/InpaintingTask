window.addEventListener("load", function onWindowLoad(){
	console.log('Загружено');
	var canvas = document.getElementById("canvas");
	var ctx = canvas.getContext("2d");
	var points = [];

	var ButDownload = document.querySelector('#ButDownload');

	 ButDownload.addEventListener('click',function(){
		 var imageCopy = document.getElementById("savedImageCopy");
		 imageCopy.src = canvas.toDataURL("image/jpeg");

		});

	// Загрузка изображения в canvas
	var img = new Image();
	img.src = 'face.jpg';
	img.addEventListener('load', function(){
		ctx.drawImage(img, 0, 0);

		var mouseX = 0;
		var mouseY = 0;

		// Стиль линии
		ctx.strokeStyle = 'red';
		ctx.lineWidth = 2;
		var isDrawing = false;

		// Обработчики рисования мышкой
		canvas.addEventListener('mousedown', function(event) {
			setMouseCoordinates(event);
			isDrawing = true;
			ctx.beginPath();
			ctx.moveTo(mouseX, mouseY);

			points.push({
				x: mouseX,
				y: mouseY,
				mode: "begin"
			});
		});

		canvas.addEventListener('mousemove', function(event) {
			setMouseCoordinates(event);
			if(isDrawing){
				ctx.lineTo(mouseX, mouseY);
				ctx.stroke();
			   	points.push({
					x: mouseX,
					y: mouseY,
					mode: "draw"
				});
			}
		});

		canvas.addEventListener('mouseup', function(event) {
			setMouseCoordinates(event);
			isDrawing = false;

			points.push({
				x: mouseX,
				y: mouseY,
				mode: "end"
			});
		});

		function setMouseCoordinates(event) {
			mouseX = event.offsetX;
			mouseY = event.offsetY;
		}

		// Кнопка «Очистить»
    var clear = document.getElementById("clear");
		clear.addEventListener('click',function(){
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			ctx.drawImage(img, 0, 0);
			return false;
		});

		// Функции для кнопки «Отменить»
		function redrawAll() {
			if (points.length == 0) {
				return;
			}
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			ctx.drawImage(img, 0, 0);
			for (var i = 0; i < points.length; i++) {
				var pt = points[i];
				if (pt.mode == "begin") {
					ctx.beginPath();
					ctx.moveTo(pt.x, pt.y);
				}
				ctx.lineTo(pt.x, pt.y);
				if (pt.mode == "end" || (i == points.length - 1)) {
					ctx.stroke();
				}
			}
			ctx.stroke();
		}

		function undoLast() {
			points.pop();
			redrawAll();
		}

		// Кнопка «Отменить»
		var interval;
    var undo = document.getElementById("undo");
    undo.addEventListener('mousedown',function(){
			interval = setInterval(undoLast, 1);
		});

    undo.addEventListener('mouseup',function(){
			clearInterval(interval);
		});

    undo.addEventListener('click',function(){
			return false;
		});

var Toolbar = document.querySelector('.Toolbar');
var imgToolbar = Toolbar.getElementsByTagName('img');

		imgToolbar[0].addEventListener('click',function(){
			ctx.lineWidth = 3;
	})
	imgToolbar[1].addEventListener('click',function(){
		ctx.lineWidth = 8;
	})
	imgToolbar[2].addEventListener('click',function(){
		ctx.lineWidth = 12;
	})

	});
});
