import React from 'react';

const Home = () => {
  return (
    <div>
      <h1>Face Recognition</h1>
      <iframe
                            src="http://127.0.0.1:5000/video_feed"
                            width="100%"
                            height="800px"
                            title="Face Recognition"
                            allowFullScreen>
                        </iframe>
    </div>
  );
};

export default Home;
