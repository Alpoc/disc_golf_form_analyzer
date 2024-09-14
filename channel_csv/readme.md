### Manually grab youtube video names and urls
1. Go to channel and select the videos tab
2. inspect and go to console
3. `allow pasting`
4. `var scroll = setInterval(function(){ window.scrollBy(0, 1000)}, 1000);`
5. `window.clearInterval(scroll); console.clear(); urls = $$('a'); urls.forEach(function(v,i,a){if (v.id=="video-title-link"){console.log('\t'+v.title+'\t'+v.href+'\t')}});`
6. copy paste into spreadsheet