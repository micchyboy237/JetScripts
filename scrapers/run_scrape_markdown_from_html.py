from jet.logger import logger
from jet.scrapers.preprocessor import scrape_markdown
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard

html_str = """
<!DOCTYPE html>
<html lang="en" prefix="og: http://ogp.me/ns#">
  <head>\n
    <meta charset="utf-8">\n
    <meta name="referrer" content="no-referrer-when-downgrade">\n
    <meta name="apple-mobile-web-app-capable" content="yes">\n
    <meta name="apple-mobile-web-app-title" content="Jobstreet">\n
    <meta name="mobile-web-app-capable" content="yes">\n
    <meta name="request-id" content="MC45ODM5ODI3MTkzMTYzODc1">\n
    <meta name="theme-color" content="#fff">\n
    <meta name="viewport" content="width=device-width, initial-scale=1">\n
    <link rel="icon" href="/static/shared-web/favicon-4e1897dfd0901e8a3bf7e604d3a90636.ico">\n
    <link rel="apple-touch-icon" href="/static/shared-web/iphone-7c4d7dcb05fece466d8901945e36bbaa.png">\n
    <link rel="apple-touch-icon" sizes="76x76" href="/static/shared-web/ipad-96988dd1c0902fd20b34ce260d03729b.png">\n
    <link rel="apple-touch-icon" sizes="120x120" href="/static/shared-web/iphoneRetina-c772d091f011ef7ce26f631360de4907.png">\n
    <link rel="apple-touch-icon" sizes="152x152" href="/static/shared-web/ipadRetina-8a06978f3bf7985f413edcb5113c9783.png">\n
    <meta name="twitter:image" content="https://ph.jobstreet.com/static/shared-web/banner-4823bc492a11077075c375b74c739042.png">\n
    <meta name="twitter:card" content="summary">\n
    <meta name="twitter:site" content="@JobstreetPH">\n
    <meta property="og:image" content="https://ph.jobstreet.com/static/shared-web/banner-4823bc492a11077075c375b74c739042.png">\n
    <meta property="og:image:width" content="1200">\n
    <meta property="og:image:height" content="1200">\n
    <meta property="og:type" content="website">\n
    <meta property="og:site_name" content="Jobstreet">\n
    <meta property="og:locale" content="en_GB">\n
    <link rel="manifest" href="/static/shared-web/manifest-3c6b41f42646e73f54ec4635ed269725.json">\n <style type="text/css">
      /* latin */
      \n@font-face {
        font-family: SeekSans;
        font-weight: 500 700;
        src: url(\'/static/shared-web/SEEKSans-Latin-Medium-0d7bb139d5772bf159e2d6ab7664bb49.woff2\') format(\'woff2\'), url(\'/static/shared-web/SEEKSans-Latin-Medium-33ee823026b461fd5f8519b9ec890974.woff\') format(\'woff\');
        font-display: swap;
        unicode-range: U+0000-00FF, U+0100-017F, U+0180-024F, U+20A0-20CF, U+02B0-02FF, U+2000-206F, U+2190-21FF, U+2122, U+FEFF, U+FFFD;
      }

      \n@font-face {
        font-family: SeekSans;
        src: url(\'/static/shared-web/SEEKSans-Latin-Regular-03b2f80a92410122dbd1e0f56109c3a0.woff2\') format(\'woff2\'), url(\'/static/shared-web/SEEKSans-Latin-Regular-e8afb4047edc1d279a35f9c327a58db2.woff\') format(\'woff\');
        font-display: swap;
        unicode-range: U+0000-00FF, U+0100-017F, U+0180-024F, U+20A0-20CF, U+02B0-02FF, U+2000-206F, U+2190-21FF, U+2122, U+FEFF, U+FFFD;
      }

      \n

      /* thai */
      \n@font-face {
        font-family: SeekSans;
        font-weight: 500 700;
        src: url(\'/static/shared-web/SEEKSans-Thai-Medium-d0f1f228aea9cf26e1b86b0c30ff4ed0.woff2\') format(\'woff2\'), url(\'/static/shared-web/SEEKSans-Thai-Medium-66b5f82427aa0e37e92cef3799f49e2a.woff\') format(\'woff\');
        font-display: swap;
        unicode-range: U+0E01-0E5B, U+200C-200D, U+25CC;
      }

      \n@font-face {
        font-family: SeekSans;
        src: url(\'/static/shared-web/SEEKSans-Thai-Regular-1ff56d415b57f8d176033fe7e7e6114a.woff2\') format(\'woff2\'), url(\'/static/shared-web/SEEKSans-Thai-Regular-d1179b08eeaec0e32a695e5d2e72bf9f.woff\') format(\'woff\');
        font-display: swap;
        unicode-range: U+0E01-0E5B, U+200C-200D, U+25CC;
      }

      \n

      /* latin - fallback */
      \n@font-face {
        font-family: "SeekSans Fallback";
        src: local(\'Arial Bold\'), local(\'Arial-BoldMT\');
        font-weight: 500 700;
        unicode-range: U+0000-00FF, U+0100-017F, U+0180-024F, U+20A0-20CF, U+02B0-02FF, U+2000-206F, U+2190-21FF, U+2122, U+FEFF, U+FFFD;
        ascent-override: 99.6737%;
        descent-override: 25.8378%;
        line-gap-override: 0%;
        size-adjust: 106.046%;
      }

      \n@font-face {
        font-family: "SeekSans Fallback";
        src: local(\'Arial\'), local(\'ArialMT\');
        unicode-range: U+0000-00FF, U+0100-017F, U+0180-024F, U+20A0-20CF, U+02B0-02FF, U+2000-206F, U+2190-21FF, U+2122, U+FEFF, U+FFFD;
        ascent-override: 93.8668%;
        descent-override: 24.3326%;
        line-gap-override: 0%;
        size-adjust: 112.6064%;
      }

      \n

      /* thai - fallback */
      \n@font-face {
        font-family: "SeekSans Fallback";
        src: local(\'Tahoma Bold\'), local(\'Tahoma-Bold\');
        font-weight: 500 700;
        unicode-range: U+0E01-0E5B, U+200C-200D, U+25CC;
        ascent-override: 103.8801%;
        descent-override: 26.9282%;
        size-adjust: 101.7519%;
      }

      \n@font-face {
        font-family: "SeekSans Fallback";
        src: local(\'Tahoma\');
        unicode-range: U+0E01-0E5B, U+200C-200D, U+25CC;
        ascent-override: 95.5394%;
        descent-override: 24.7661%;
        size-adjust: 110.635%;
      }
    </style>\n \n
    <link rel="preconnect" href="https://bx-branding-gateway.cloud.seek.com.au">\n
    <link rel="preconnect" href="https://image-service-cdn.seek.com.au">\n
    <link rel="preconnect" href="https://web.aips-sol.com">\n
    <link rel="preconnect" href="https://cdn.seeklearning.com.au">\n
    <link rel="dns-prefetch" href="https://bx-branding-gateway.cloud.seek.com.au">\n
    <link rel="dns-prefetch" href="https://image-service-cdn.seek.com.au">\n
    <link rel="dns-prefetch" href="https://web.aips-sol.com">\n
    <link rel="dns-prefetch" href="https://cdn.seeklearning.com.au">\n\n
    <link data-chunk="main" rel="stylesheet" href="/static/ca-search-ui/houston/vendor.braid-882bc9f22a3e5504306b.css">\n
    <link data-chunk="main" rel="stylesheet" href="/static/ca-search-ui/houston/app-21d259fa9866c02128a4.css">\n <script>
      \
      n(function(l, d, j, t) {
            \
            n d.test(l.hash) && j.test(l.pathname) && (l.href = l.pathname + (l.search ? l.search + \'&\' : \'?\') + l.hash.slice(1) + t);\n    }(location,/^#daterange=/i,/^\\/jobs/,\'&hashredirect=true\'))\n  
    </script>\n
    <meta property="fb:app_id" content="160055201197039">\n \n <title data-rh="true">Front End Developer Job in Manila City at CoDev - Jobstreet</title>
    <meta data-rh="true" name="robots" content="index,follow">
    <meta data-rh="true" property="og:url" content="https://ph.jobstreet.com/job/82411354">
    <meta data-rh="true" property="og:title" content="Front End Developer Job in Manila City at CoDev - Jobstreet">
    <meta data-rh="true" property="twitter:title" content="Front End Developer Job in Manila City at CoDev - Jobstreet">
    <meta data-rh="true" name="description" content="Will work on a web-based application that allows users to input metropolitan areas and healthcare billing codes">
    <meta data-rh="true" property="og:description" content="Will work on a web-based application that allows users to input metropolitan areas and healthcare billing codes">
    <meta data-rh="true" property="twitter:description" content="Will work on a web-based application that allows users to input metropolitan areas and healthcare billing codes">
    <link data-rh="true" rel="canonical" href="https://ph.jobstreet.com/job/82411354">
    <script data-rh="true" type="application/ld+json">
      {
        "@context": "http://schema.org",
        "@graph": [{
          "@type": "WebSite",
          "url": "https://ph.jobstreet.com",
          "potentialAction": {
            "@type": "SearchAction",
            "target": "https://ph.jobstreet.com/{search_term_string}-jobs",
            "query-input": "required name=search_term_string"
          }
        }]
      }
    </script>
    <script data-rh="true" type="application/ld+json">
      {
        "@context": "https://schema.org/",
        "@type": "JobPosting",
        "datePosted": "2025-02-27T08:15:14.991Z",
        "description": " < p > The Front - End Developer will work on the < strong > healthcare price benchmarking tool < /strong>, a web-based application that allows users to input metropolitan areas and healthcare billing codes (CPT codes) to receive relevant benchmark pricing data. This tool will include: < /p> < ul > < li > A < strong > searchable interface < /strong> for users to input and retrieve pricing data < /li> < li > < strong > Data visualization elements < /strong> such as percentile ranges, regional heat maps, and interactive graphs < /li> < li > < strong > Export functionality < /strong> for reports and datasets < /li> < /ul> < p > The back - end,
        including data scrubbing and storage,
        will be handled by Simple Healthcare,
        with the front - end focused on presenting and visualizing this information. < /p> < p > This is the first step in a broader roadmap,
        with a more < strong > complex BI(Business Intelligence) dashboard < /strong> planned for future development in 2026. < /p> < p > < strong > Responsibilities < /strong> < /p> < ul > < li > Develop a < strong > scalable,
        user - friendly front - end < /strong> for the price benchmarking tool < /li> < li > Implement < strong > search and filtering functionalities < /strong> for healthcare pricing data < /li> < li > Develop < strong > data visualization elements < /strong> (heat maps, percentile charts, interactive tables) < /li> < li > Optimize front - end performance
        for < strong > speed and accessibility < /strong> < /li> < li > Collaborate with a < strong > UI / UX designer < /strong> to implement user-friendly designs < /li> < li > Ensure the front - end integrates seamlessly with the back - end APIs < /li> < li > Maintain < strong > clean,
        scalable,
        and well - documented < /strong> code for future product expansion < /li> < li > Work closely with stakeholders to gather and refine requirements < /li> < /ul> < p > < strong > Qualifications < /strong> < /p> < p > < strong > Must - Have: < /strong> < /p> < ul > < li > < strong > 4 + years < /strong> of front-end development experience < /li> < li > Strong proficiency in < strong > React.js < /strong> (preferred) or  < strong > Vue.js < /strong> < /li> < li > Experience with < strong > Next.js < /strong> (or similar SSR frameworks) < /li> < li > Strong knowledge of < strong > JavaScript,
        TypeScript,
        HTML5,
        CSS3 < /strong> < /li> < li > Experience with < strong > RESTful APIs and JSON data handling < /strong> < /li> < li > Familiarity with < strong > data visualization libraries < /strong> (D3.js, Chart.js, Highcharts, or similar) < /li> < li > Knowledge of < strong > state management < /strong> (Redux, Context API) < /li> < li > Understanding of < strong > responsive and accessible web design < /strong> < /li> < /ul> < p > < strong > Nice - to - Have: < /strong> < /p> < ul > < li > Experience with < strong > BI tools < /strong> or  < strong > data - heavy applications < /strong> < /li> < li > Familiarity with < strong > Material UI,
        Tailwind CSS,
        or other UI libraries < /strong> < /li> < li > Experience with < strong > healthcare or financial tech applications < /strong> < /li> < li > Basic understanding of < strong > backend technologies(Node.js, Express.js, PostgreSQL / MongoDB) < /strong> < /li> < li > Exposure to < strong > React Native < /strong> for potential future mobile development < /li> < /ul> < p > < strong > Additional Skill / Ideal: < /strong> < /p> < ul > < li > Basic knowledge of < strong > SQL databases < /strong>. < /li> < li > Has understanding of < strong > UI / UX principles < /strong> and front-end performance optimization. < /li> < /ul>","hiringOrganization":{"@type":"Organization","name":"Complete Development (CoDev)","sameAs":"https:/ / www.codev.com / careers / ","
        logo ":"
        https: //image-service-cdn.seek.com.au/fce76906ed15a97d1c1f92c5b6b3125153f11c0e/ee4dce1061f3f616224767ad58cb2fc751b8d2dc"},"jobLocation":{"@type":"Place","address":{"@type":"PostalAddress","addressLocality":"Metro Manila","addressCountry":"Philippines"}},"title":"Mid-Level Frontend Developer","directApply":true,"employmentType":["FULL_TIME"],"identifier":{"@type":"PropertyValue","name":"Complete Development (CoDev)","value":"82411354"},"validThrough":"2025-03-29T12:59:59.999Z"}
    </script>
  </head>\n <body style="overflow-x:hidden">\n <div id="app">
      <style type="text/css">
        \n html,
        body {
          margin: 0;
          padding: 0;
          background: #fff
        }

        \n html.eihuid10,
        html.eihuid10 body {
          color-scheme: dark;
          background: #1C2330
        }

        \n
      </style>
      <div class="k73btx0 i7p5ej18 i7p5ej1b">
        <span class="gepq850 im98kj2 im98kj3 i7p5ej2 eihuidh" role="button" tabindex="0" data-automation="g-1-t-btn"></span>
        <meta data-automation="deeplink-metadata-preview" name="branch:deeplink:preview" content="true">
        <meta data-automation="deeplink-metadata-previewName" name="branch:deeplink:previewName" content="OTHER">
        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
          <div class="gepq850 _1r155bc0">
            <span class="gepq850 im98kj2 im98kj4 im98kj7 i7p5ej2 eihuidh" role="button" tabindex="0">Skip to content</span>
          </div>
        </span>
        <div class="gepq850 eihuid5j eihuid0 _1qnq8v60">
          <span class="gepq850 eihuid5" aria-live="polite" data-automation="screen-reader-announcer" id="screen-reader-announcer" role="status"></span>
        </div>
        <div data-focus-guard="true" tabindex="-1" style="width:1px;height:0px;padding:0;overflow:hidden;position:fixed;top:1px;left:1px"></div>
        <div data-focus-lock-disabled="disabled">
          <header class="gepq850 gepq851">
            <div class="gepq850 eihuid8j eihuid7s eihuid9n eihuid8w eihuid5b eihuidhf eihuidgv eihuidp eihuid5g i7p5ej18 i7p5ej1b eihuid33 eihuid36">
              <div class="gepq850 eihuidbv eihuidaw eihuidar eihuid9s">
                <div class="gepq850 eihuidp eihuidv gfmryc0">
                  <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgr eihuidh4">
                    <div class="gepq850 eihuid5b eihuidhf eihuidh8">
                      <div class="gepq850 eihuid7v eihuid8k eihuid8z eihuid9o eihuidb3 eihuidbw eihuid9z eihuidas eihuid5b eihuidgj eihuidh3 eihuidgs eihuides egznas1 i7p5ej18 i7p5ej1b eihuid33 eihuid36">
                        <a href="/" class="gepq850 gepq85f  gepq850 gepq85f _19e8v9v0" data-automation="company logo" target="_self">
                          <div class="gepq850 eihuid5j eihuid0 _1qnq8v60" data-automation="jobstreet">Jobstreet</div>
                          <span class="gepq850 eihuid4z">
                            <svg viewBox="0 0 248 66" height="40" class="eihuid4z ysqkjq9 _1t7re5j0">
                              <circle cy="32.98" cx="32.98" r="30" fill="#fff" class="ysqkjqe"></circle>
                              <mask id="jobStreetCutArrowOutOfCircle">
                                <circle fill="white" cx="32.98" cy="32.98" r="32.98"></circle>
                                <path fill="black" d="M33.76 12.58c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06M40.18 19.51c0-1.26 1.02-2.28 2.27-2.28s2.28 1.02 2.28 2.28-1.02 2.27-2.28 2.27-2.27-1.02-2.27-2.27M33.76 19.51c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.93-2.06-2.06M47 26.46c0-1.41 1.14-2.55 2.55-2.55s2.55 1.14 2.55 2.55-1.14 2.55-2.55 2.55S47 27.87 47 26.46M40.18 26.44c0-1.26 1.02-2.27 2.27-2.27s2.28 1.02 2.28 2.27-1.02 2.27-2.28 2.27-2.27-1.02-2.27-2.27M33.76 26.44c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06M27.64 26.44c0-1 .81-1.8 1.8-1.8s1.81.81 1.81 1.8-.81 1.8-1.81 1.8-1.8-.81-1.8-1.8M22.53 26.44c0-.85.69-1.55 1.54-1.55s1.54.69 1.54 1.55-.69 1.55-1.54 1.55-1.54-.69-1.54-1.55M17.66 26.44c0-.71.58-1.29 1.29-1.29s1.29.58 1.29 1.29-.57 1.29-1.29 1.29-1.29-.58-1.29-1.29M13.53 26.44c0-.57.46-1.03 1.03-1.03s1.03.46 1.03 1.03-.46 1.03-1.03 1.03-1.03-.46-1.03-1.03M9.63 26.44c0-.43.34-.77.77-.77s.77.35.77.77-.35.77-.77.77-.77-.35-.77-.77M6.33 26.44c0-.29.23-.51.52-.51s.51.23.51.51-.23.52-.51.52-.52-.23-.52-.52M47 33.39c0-1.41 1.14-2.55 2.55-2.55s2.55 1.15 2.55 2.55-1.14 2.55-2.55 2.55S47 34.8 47 33.39M40.18 33.37c0-1.26 1.02-2.27 2.27-2.27s2.28 1.01 2.28 2.27-1.02 2.28-2.28 2.28-2.27-1.02-2.27-2.28M33.76 33.37c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06M27.64 33.37c0-1 .81-1.8 1.8-1.8s1.81.81 1.81 1.8-.81 1.8-1.81 1.8-1.8-.81-1.8-1.8M22.53 33.37c0-.85.69-1.55 1.54-1.55s1.54.69 1.54 1.55-.69 1.55-1.54 1.55-1.54-.69-1.54-1.55M17.66 33.37c0-.71.58-1.29 1.29-1.29s1.29.57 1.29 1.29-.57 1.29-1.29 1.29-1.29-.58-1.29-1.29M13.53 33.37c0-.57.46-1.03 1.03-1.03s1.03.46 1.03 1.03-.46 1.03-1.03 1.03-1.03-.46-1.03-1.03M9.63 33.37c0-.43.34-.77.77-.77s.77.35.77.77-.35.77-.77.77-.77-.34-.77-.77M6.33 33.37c0-.29.23-.52.52-.52s.51.23.51.52-.23.52-.51.52-.52-.23-.52-.52M54 33.44c0-1.55 1.26-2.8 2.8-2.8s2.8 1.25 2.8 2.8-1.25 2.79-2.8 2.79-2.8-1.25-2.8-2.79M47 40.32c0-1.41 1.14-2.55 2.55-2.55s2.55 1.14 2.55 2.55-1.14 2.55-2.55 2.55S47 41.73 47 40.32M40.18 40.3c0-1.26 1.02-2.28 2.27-2.28s2.28 1.02 2.28 2.28-1.02 2.27-2.28 2.27-2.27-1.02-2.27-2.27M33.76 40.3c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06M27.64 40.3c0-1 .81-1.81 1.8-1.81s1.81.81 1.81 1.81-.81 1.8-1.81 1.8-1.8-.8-1.8-1.8M22.53 40.3c0-.86.69-1.55 1.54-1.55s1.54.69 1.54 1.55-.69 1.54-1.54 1.54-1.54-.69-1.54-1.54M17.66 40.3c0-.72.58-1.29 1.29-1.29s1.29.57 1.29 1.29-.57 1.29-1.29 1.29-1.29-.58-1.29-1.29M13.53 40.3c0-.57.46-1.03 1.03-1.03s1.03.46 1.03 1.03-.46 1.03-1.03 1.03-1.03-.46-1.03-1.03M9.63 40.3c0-.43.34-.78.77-.78s.77.35.77.78-.35.77-.77.77-.77-.35-.77-.77M6.33 40.3c0-.29.23-.52.52-.52s.51.23.51.52-.23.51-.51.51-.52-.23-.52-.51M40.18 47.23c0-1.26 1.02-2.28 2.27-2.28s2.28 1.02 2.28 2.28-1.02 2.27-2.28 2.27-2.27-1.02-2.27-2.27M33.76 47.23c0-1.14.92-2.07 2.06-2.07s2.06.93 2.06 2.07-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06M33.76 54.16c0-1.14.92-2.06 2.06-2.06s2.06.92 2.06 2.06-.92 2.06-2.06 2.06-2.06-.92-2.06-2.06"></path>
                              </mask>
                              <circle fill="#0d3880" class="ysqkjqd" cx="32.98" cy="32.98" r="32.98" mask="url(#jobStreetCutArrowOutOfCircle)"></circle>
                              <path fill="#000" class="ysqkjqf" d="M82.79 17.04h-5.98V12.2h5.98v4.84Zm0 29.92c0 1.86-.55 3.41-1.64 4.66-1.25 1.43-3.01 2.15-5.3 2.15h-3.38v-5.02h2.26c1.39 0 2.08-.72 2.08-2.15V21.07h5.98v25.9ZM100.97 32.94c0-2.92-.45-4.84-1.36-5.76-.69-.7-1.61-1.05-2.76-1.05s-2.02.35-2.71 1.05c-.9.91-1.36 2.83-1.36 5.76s.45 4.89 1.36 5.8c.69.7 1.6 1.05 2.71 1.05s2.06-.35 2.76-1.05c.9-.91 1.36-2.85 1.36-5.8m5.98 0c0 2.28-.18 4.1-.55 5.44-.4 1.49-1.11 2.77-2.15 3.84-1.86 1.95-4.32 2.92-7.4 2.92s-5.5-.97-7.35-2.92c-1.04-1.07-1.75-2.34-2.15-3.84-.37-1.34-.55-3.15-.55-5.44 0-4.26.91-7.35 2.74-9.27s4.26-2.88 7.31-2.88 5.53.96 7.35 2.88c1.83 1.92 2.74 5.01 2.74 9.27M124.99 32.94c0-2.1-.17-3.61-.5-4.52-.6-1.52-1.76-2.28-3.48-2.28s-2.88.76-3.48 2.28c-.33.91-.5 2.42-.5 4.52s.17 3.61.5 4.52c.6 1.55 1.76 2.33 3.48 2.33s2.87-.78 3.48-2.33c.33-.91.5-2.42.5-4.52m5.98 0c0 2.44-.11 4.26-.32 5.48-.34 1.98-1.04 3.5-2.1 4.57-1.43 1.43-3.37 2.15-5.8 2.15s-4.42-.84-5.94-2.51v2.24h-5.76V12.34h5.98v10.83c1.43-1.58 3.34-2.37 5.74-2.37s4.36.72 5.78 2.15c1.06 1.07 1.76 2.59 2.09 4.57.21 1.22.32 3.03.32 5.44M153.04 37.37c0 2.53-.98 4.48-2.92 5.85-1.83 1.28-4.22 1.92-7.17 1.92-2.22 0-4.04-.2-5.44-.59-1.77-.52-3.33-1.46-4.71-2.83l3.88-3.88c1.49 1.49 3.61 2.24 6.35 2.24s4.2-.82 4.2-2.47c0-1.31-.84-2.04-2.51-2.19l-3.75-.37c-4.63-.46-6.94-2.68-6.94-6.67 0-2.37.93-4.26 2.79-5.66 1.7-1.28 3.84-1.92 6.39-1.92 4.08 0 7.11.93 9.09 2.79l-3.65 3.7c-1.19-1.07-3.03-1.6-5.53-1.6-2.25 0-3.38.76-3.38 2.28 0 1.22.82 1.9 2.47 2.06l3.75.37c4.72.46 7.08 2.79 7.08 6.99M167.16 44.86h-3.24c-2.25 0-4-.72-5.25-2.15-1.1-1.25-1.64-2.8-1.64-4.66V26.26h-2.51v-4.52h2.51v-7.03h5.98v7.03h4.16v4.52h-4.16v11.42c0 1.43.68 2.15 2.03 2.15h2.12v5.02ZM188.35 23.02l-4.48 4.52c-.94-.94-1.99-1.42-3.15-1.42-1.01 0-1.87.35-2.6 1.05-.82.82-1.23 1.93-1.23 3.33v14.34h-5.94v-23.8h5.8v2.28c1.43-1.7 3.43-2.56 5.98-2.56 2.25 0 4.13.75 5.62 2.24M203.88 30.74c-.03-.97-.21-1.83-.55-2.56-.73-1.64-2.06-2.47-3.97-2.47s-3.24.82-3.97 2.47c-.34.73-.52 1.58-.55 2.56h9.04Zm5.85 4.07h-14.89c0 1.58.46 2.86 1.39 3.84.93.97 2.2 1.46 3.81 1.46 2.1 0 3.9-.75 5.39-2.24l3.61 3.52c-1.31 1.31-2.59 2.24-3.84 2.79-1.43.64-3.17.96-5.21.96-7.34 0-11.01-4.07-11.01-12.2 0-3.81.96-6.81 2.88-9 1.86-2.1 4.35-3.15 7.49-3.15s5.79 1.08 7.67 3.24c1.8 2.07 2.69 4.78 2.69 8.13v2.65ZM227.36 30.74c-.03-.97-.21-1.83-.55-2.56-.73-1.64-2.06-2.47-3.97-2.47s-3.24.82-3.97 2.47c-.34.73-.52 1.58-.55 2.56h9.04Zm5.85 4.07h-14.89c0 1.58.46 2.86 1.39 3.84s2.2 1.46 3.81 1.46c2.1 0 3.9-.75 5.39-2.24l3.61 3.52c-1.31 1.31-2.59 2.24-3.84 2.79-1.43.64-3.17.96-5.21.96-7.34 0-11.01-4.07-11.01-12.2 0-3.81.96-6.81 2.88-9 1.86-2.1 4.35-3.15 7.49-3.15s5.79 1.08 7.67 3.24c1.8 2.07 2.69 4.78 2.69 8.13v2.65ZM247.87 44.86h-3.24c-2.25 0-4-.72-5.25-2.15-1.1-1.25-1.64-2.8-1.64-4.66V26.26h-2.51v-4.52h2.51v-7.03h5.98v7.03h4.16v4.52h-4.16v11.42c0 1.43.68 2.15 2.03 2.15h2.12v5.02ZM201.26 56.85c0-.75-.06-1.29-.18-1.62-.22-.55-.63-.82-1.25-.82s-1.03.27-1.25.82c-.12.33-.18.87-.18 1.62s.06 1.29.18 1.62c.22.56.63.83 1.25.83s1.03-.28 1.25-.83c.12-.33.18-.87.18-1.62m2.14 0c0 .87-.04 1.53-.11 1.96-.12.71-.37 1.25-.75 1.64-.51.51-1.21.77-2.08.77s-1.58-.3-2.13-.9v.8h-2.06V49.47h2.14v3.88c.51-.57 1.2-.85 2.06-.85s1.56.26 2.07.77c.38.38.63.93.75 1.64.08.44.11 1.09.11 1.95M211.7 52.59l-3.65 9.9c-.17.48-.39.85-.64 1.1-.44.44-1.05.65-1.83.65h-.83v-1.91h.49c.32 0 .55-.06.7-.16.15-.11.27-.31.37-.61l.36-1.05-2.91-7.92H206l1.77 5.27 1.7-5.27h2.24ZM222.08 58.43c0 .91-.35 1.6-1.05 2.09-.66.46-1.51.69-2.57.69-.8 0-1.45-.07-1.95-.21-.63-.19-1.2-.52-1.69-1.01l1.39-1.39c.53.54 1.29.8 2.28.8s1.5-.29 1.5-.88c0-.47-.3-.73-.9-.79l-1.34-.13c-1.66-.16-2.49-.96-2.49-2.39 0-.85.33-1.53 1-2.03.61-.46 1.37-.69 2.29-.69 1.46 0 2.55.33 3.26 1l-1.31 1.33c-.43-.38-1.09-.57-1.98-.57-.81 0-1.21.27-1.21.82 0 .44.29.68.88.74l1.34.13c1.69.16 2.54 1 2.54 2.5M228.2 56.06c-.01-.35-.08-.65-.2-.92-.26-.59-.74-.88-1.42-.88s-1.16.29-1.42.88c-.12.26-.19.57-.2.92h3.24Zm2.09 1.46h-5.34c0 .57.17 1.03.5 1.37.33.35.79.52 1.37.52.75 0 1.4-.27 1.93-.8l1.29 1.26c-.47.47-.93.8-1.37 1-.51.23-1.14.34-1.87.34-2.63 0-3.94-1.46-3.94-4.37 0-1.36.34-2.44 1.03-3.22.67-.75 1.56-1.13 2.68-1.13s2.07.39 2.75 1.16c.64.74.97 1.71.97 2.91v.95ZM236.61 56.06c-.01-.35-.08-.65-.2-.92-.26-.59-.74-.88-1.42-.88s-1.16.29-1.42.88c-.12.26-.19.57-.2.92h3.24Zm2.09 1.46h-5.34c0 .57.17 1.03.5 1.37.33.35.79.52 1.37.52.75 0 1.4-.27 1.93-.8l1.29 1.26c-.47.47-.93.8-1.37 1-.51.23-1.14.34-1.87.34-2.63 0-3.94-1.46-3.94-4.37 0-1.36.34-2.44 1.03-3.22.67-.75 1.56-1.13 2.68-1.13s2.07.39 2.75 1.16c.64.74.97 1.71.97 2.91v.95ZM247.87 61.12h-2.63l-2.09-3.55-.9 1.01v2.54h-2.14V49.47h2.14v6.61l2.83-3.49h2.56l-3.05 3.44 3.28 5.09z"></path>
                            </svg>
                          </span>
                        </a>
                        <div class="gepq850 eihuid4z eihuid4w">
                          <button class="gepq850 gepq857 eihuid5b eihuidh" data-automation="Mobile Menu" aria-expanded="false">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidaz">Menu</div>
                                <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                      <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                    </svg>
                                  </span>
                                </div>
                              </div>
                            </span>
                          </button>
                        </div>
                      </div>
                      <div class="gepq850 eihuid4v eihuid5d">
                        <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgf eihuidgk _18isxyp2">
                          <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                            <a href="/" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f _18isxyp3" data-automation="job search" data-automation-role="nav-tabs" target="_self">
                              <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 _18isxyp9 _18isxypa" data-title="Job search">Job search</span>
                                </span>
                                <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6 _18isxyp7"></span>
                              </span>
                            </a>
                          </div>
                          <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                            <a href="/profile" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="profile" data-automation-role="nav-tabs" target="_self">
                              <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 _18isxyp9" data-title="Profile">Profile</span>
                                </span>
                                <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                              </span>
                            </a>
                          </div>
                          <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                            <a href="/career-advice" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="career advice" data-automation-role="nav-tabs" target="_self">
                              <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 _18isxyp9" data-title="Career advice">Career advice</span>
                                </span>
                                <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                              </span>
                            </a>
                          </div>
                          <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                            <a href="/companies" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="company reviews" data-automation-role="nav-tabs" target="_self">
                              <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 _18isxyp9" data-title="Explore companies">Explore companies</span>
                                </span>
                                <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                              </span>
                            </a>
                          </div>
                          <div class="gepq850 eihuidar _18isxyp2">
                            <a href="/community/" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="community" data-automation-role="nav-tabs" target="_self">
                              <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidef eihuidb7 eihuidbw">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 _18isxyp9" data-title="Community">Community</span>
                                </span>
                                <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                              </span>
                              <div class="gepq850 _14schll1 eihuid5f">
                                <span class="gepq850 eihuid4z _16qi62m0 _16qi62m3 _16qi62m1 _16qi62m17">
                                  <span class="gepq850 eihuid4z eihuid5f">
                                    <span class="gepq850 eihuid7n eihuid8r eihuidb3 eihuid9z eihuid63 eihuid0 eihuidg eihuidw eihuid5b i7p5ej18 i7p5ej1b i7p5ej1i i7p5ej1r eihuid2x eihuid2i" title="New">
                                      <span class="gepq850 eihuid4z eihuidr i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5ej4 _1q03wcw0">
                                        <span class="gepq850 _1q03wcw2 eihuid4z eihuid0 eihuidr">New</span>
                                      </span>
                                    </span>
                                  </span>
                                </span>
                              </div>
                            </a>
                          </div>
                        </div>
                      </div>
                      <div class="gepq850 eihuid4v eihuid4w eihuidi3 vd354n0 vd354n1 i7p5ej18 i7p5ej1b eihuid11 eihuid14">
                        <div class="gepq850 eihuid8v i7p5ej18 i7p5ej1b eihuid33 eihuid36">
                          <div class="gepq850 eihuid7r eihuid8v">
                            <span class="gepq850 eihuid4z eihuid5f eihuidp">
                              <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
                            </span>
                          </div>
                          <div class="gepq850 eihuid8z eihuid7v eihuidb7 eihuida3">
                            <div class="gepq850 eihuidn eihuid5b _1l3fjc41" data-automation="sign-in-register">
                              <div class="gepq850 eihuidn eihuidhr eihuid5b eihuidgj">
                                <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6r eihuidhn _5qpagp0">
                                  <a href="/oauth/login/?returnUrl=http%3A%2F%2Fph.jobstreet.com%2Fjob%2F82411354" rel="nofollow" class="gepq850 gepq85f  gepq850 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka7 f9bd8p0" title="Sign in" data-automation="sign in" target="_self">
                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1b eihuid2b eihuid2m"></span>
                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1b eihuid29 eihuid2k"></span>
                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid63 eihuid5j eihuidi eihuid3z eihuid44"></span>
                                    <span class="gepq850 eihuidb7 eihuida3 eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                      <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5ej7">Sign in</span>
                                    </span>
                                  </a>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div class="gepq850 eihuid7r eihuid8v">
                            <span class="gepq850 eihuid4z eihuid5f eihuidp">
                              <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
                            </span>
                          </div>
                          <button class="gepq850 gepq857 eihuidb7 eihuida3 eihuid4z eihuidp eihuidh _4fdslw0" data-automation="country" aria-expanded="false">
                            <div class="gepq850 eihuid7z eihuid93">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                <div class="gepq850 eihuidgj eihuid5b eihuidh3">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuid6j eihuidhn _5qpagp0">
                                    <span class="gepq850 eihuid57">
                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                        <path d="M12 1C7.6 1 4 4.6 4 9c0 4.1 6.5 12.6 7.2 13.6.2.2.5.4.8.4s.6-.1.8-.4c.7-1 7.2-9.5 7.2-13.6 0-4.4-3.6-8-8-8zm0 19.3c-2.2-3-6-8.8-6-11.3 0-3.3 2.7-6 6-6s6 2.7 6 6c0 2.5-3.8 8.3-6 11.3z"></path>
                                        <path d="M12 5c-1.7 0-3 1.3-3 3s1.3 3 3 3 3-1.3 3-3-1.3-3-3-3zm0 4c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1z"></path>
                                      </svg>
                                    </span>Philippines
                                  </div>
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                      <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                    </svg>
                                  </span>
                                </div>
                              </span>
                            </div>
                          </button>
                          <a href="https://www.seek.com.au/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Australia" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Australia
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://hk.jobsdb.com/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Hong Kong" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Hong Kong
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://id.jobstreet.com/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Indonesia" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Indonesia
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://my.jobstreet.com/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Malaysia" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Malaysia
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://www.seek.co.nz/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - New Zealand" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>New Zealand
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Philippines" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej3 i7p5ej21 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Philippines
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://sg.jobstreet.com/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Singapore" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Singapore
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <a href="https://th.jobsdb.com/" class="gepq850 gepq85f  gepq850 gepq85f eihuid4v" data-automation="Country - Thailand" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidbf eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">
                                    <div class="gepq850 _9sbqkh3">
                                      <span class="gepq850 eihuid57">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                          <path d="M19.7 6.3c-.4-.4-1-.4-1.4 0L9 15.6l-3.3-3.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l4 4c.2.2.4.3.7.3s.5-.1.7-.3l10-10c.4-.4.4-1 0-1.4z"></path>
                                        </svg>
                                      </span>
                                    </div>Thailand
                                  </div>
                                </span>
                              </div>
                            </div>
                          </a>
                          <div class="gepq850 eihuid7r eihuid8v">
                            <span class="gepq850 eihuid4z eihuid5f eihuidp">
                              <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
                            </span>
                          </div>
                          <a href="https://ph.employer.seek.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _9sbqkh0" data-automation="employer site | for employers" target="_self">
                            <div class="gepq850 eihuid7z eihuid93 eihuidh eihuid5b eihuidb7 eihuidgj">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5eja">
                                  <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0">Employer site</div>
                                </span>
                              </div>
                            </div>
                          </a>
                        </div>
                        <span class="gepq850 eihuid4z eihuid5f eihuidp">
                          <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
                        </span>
                      </div>
                    </div>
                    <div class="gepq850 eihuid4v eihuid5c">
                      <div class="gepq850 eihuidfr" data-automation="desktop-auth-links-wrapper">
                        <div class="gepq850 eihuidn eihuid5b _1l3fjc41" data-automation="sign-in-register">
                          <div class="gepq850 eihuidn eihuidhr eihuid5b eihuidgj">
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6r eihuidhn _5qpagp0">
                              <a href="/oauth/login/?returnUrl=http%3A%2F%2Fph.jobstreet.com%2Fjob%2F82411354" rel="nofollow" class="gepq850 gepq85f  gepq850 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka7 f9bd8p0" title="Sign in" data-automation="sign in" target="_self">
                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1b eihuid2b eihuid2m"></span>
                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1b eihuid29 eihuid2k"></span>
                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid63 eihuid5j eihuidi eihuid3z eihuid44"></span>
                                <span class="gepq850 eihuidb7 eihuida3 eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                  <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5ej7">Sign in</span>
                                </span>
                              </a>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="gepq850 eihuidfr">
                        <div class="gepq850 eihuid5b eihuidgj eihuidn">
                          <a href="https://ph.employer.seek.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _1vpfj3e0" data-automation="employers_link" target="_self">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5ej7">Employer site</span>
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div class="gepq850 eihuid4v eihuid50 eihuid4x eihuid7n">
                    <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgf eihuidgk _18isxyp2">
                      <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                        <a href="/" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f _18isxyp3" data-automation="job search" data-automation-role="nav-tabs" target="_self">
                          <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5ej7">
                              <span class="gepq850 _18isxyp9 _18isxypa" data-title="Job search">Job search</span>
                            </span>
                            <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6 _18isxyp7"></span>
                          </span>
                        </a>
                      </div>
                      <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                        <a href="/profile" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="profile" data-automation-role="nav-tabs" target="_self">
                          <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                              <span class="gepq850 _18isxyp9" data-title="Profile">Profile</span>
                            </span>
                            <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                          </span>
                        </a>
                      </div>
                      <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                        <a href="/career-advice" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="career advice" data-automation-role="nav-tabs" target="_self">
                          <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                              <span class="gepq850 _18isxyp9" data-title="Career advice">Career advice</span>
                            </span>
                            <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                          </span>
                        </a>
                      </div>
                      <div class="gepq850 eihuidar eihuid9s _18isxyp2">
                        <a href="/companies" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="company reviews" data-automation-role="nav-tabs" target="_self">
                          <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidf7 eihuidb7 eihuidbw">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                              <span class="gepq850 _18isxyp9" data-title="Explore companies">Explore companies</span>
                            </span>
                            <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                          </span>
                        </a>
                      </div>
                      <div class="gepq850 eihuidar _18isxyp2">
                        <a href="/community/" class="gepq850 gepq85f  gepq850 gepq85f _18isxyp1 gepq850 gepq85f eihuid5b eihuidgj eihuidh eihuid0 eihuid5f" data-automation="community" data-automation-role="nav-tabs" target="_self">
                          <span class="gepq850 eihuid7z eihuid8k eihuid93 eihuid9o eihuid5b eihuidgj eihuidn eihuid5f eihuidef eihuidb7 eihuidbw">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                              <span class="gepq850 _18isxyp9" data-title="Community">Community</span>
                            </span>
                            <span class="gepq850 eihuid4v eihuid54 _18isxyp6 eihuid5j eihuidp eihuidx eihuid63 eihuidcj eihuid6"></span>
                          </span>
                          <div class="gepq850 _14schll1 eihuid5f">
                            <span class="gepq850 eihuid4z _16qi62m0 _16qi62m3 _16qi62m1 _16qi62m17">
                              <span class="gepq850 eihuid4z eihuid5f">
                                <span class="gepq850 eihuid7n eihuid8r eihuidb3 eihuid9z eihuid63 eihuid0 eihuidg eihuidw eihuid5b i7p5ej18 i7p5ej1b i7p5ej1i i7p5ej1r eihuid2x eihuid2i" title="New">
                                  <span class="gepq850 eihuid4z eihuidr i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5ej4 _1q03wcw0">
                                    <span class="gepq850 _1q03wcw2 eihuid4z eihuid0 eihuidr">New</span>
                                  </span>
                                </span>
                              </span>
                            </span>
                          </div>
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </header>
          <span class="gepq850 eihuid4z eihuid5f eihuidp">
            <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
          </span>
        </div>
        <div data-focus-guard="true" tabindex="-1" style="width:1px;height:0px;padding:0;overflow:hidden;position:fixed;top:1px;left:1px"></div>
        <div role="main">
          <div class="gepq850 eihuid85" data-automation="jobDetailsPage">
            <div class="gepq850 eihuidp eihuidu gfmryc0">
              <div class="gepq850 eihuid5" tabindex="-1" id="start-of-content"></div>
              <div class="gepq850 eihuid97 i7p5ej18 i7p5ej1b eihuid33 eihuid36">
                <div class="gepq850 eihuidc eihuidj eihuid2 u4vyxwc">
                  <div class="gepq850 eihuid4z _17mt9yf0"></div>
                </div>
                <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                  <div class="gepq850 eihuid0 eihuid6d u4vyxwd">
                    <div class="gepq850 eihuid5f">
                      <div class="gepq850 eihuid5f eihuid0 _12zol7x2 i7p5ej18 i7p5ej1b eihuid2n eihuid2i" data-testid="bx-cover-container">
                        <div class="gepq850 eihuid5b eihuidhf eihuidgj eihuidgv _12zol7x8 eihuid5j eihuidj eihuidl eihuidp eihuidn _12zol7x6">
                          <div data-testid="bx-cover-image" class="_12zol7xj"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div class="gepq850 eihuidb3 eihuidaw eihuidbl eihuid9z eihuid9s eihuidah">
                    <div class="gepq850 eihuid5b eihuidhf eihuid73">
                      <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                        <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                          <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2d">
                            <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidb7">
                              <div class="gepq850 eihuidn eihuid5f _6xfl992" data-testid="bx-logo-container">
                                <div class="gepq850 eihuid5b eihuidgj eihuidn">
                                  <div class="gepq850 eihuidp eihuidn i7p5ej18 i7p5ej1b eihuid33 eihuid36">
                                    <div class="gepq850 eihuid5b eihuidn" data-testid="bx-logo-image"></div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb7">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid7f eihuidhn _5qpagp0">
                                <div class="gepq850">
                                  <div class="gepq850 eihuid5f">
                                    <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2c">
                                      <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb3">
                                        <div class="gepq850">
                                          <button class="gepq850 gepq857 eihuid8v eihuid7r eihuidb3 eihuid9z eihuid7 eihuidw eihuid5v eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygkah vqygka0 vqygka6 _1ucsmu70 f9bd8p0 i7p5ej18 i7p5ej1b eihuid2p" type="button" id="OptionsDropdownMenuButton" aria-label="Share or report ad" aria-haspopup="true" aria-expanded="false" tabindex="0" data-automation="option-drop-down-menu">
                                            <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid5v eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                            <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid5v eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1a eihuid2t eihuid2u"></span>
                                            <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid5v eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1a eihuid2r eihuid2s"></span>
                                            <span class="gepq850 eihuid4z eihuid5f _74wkf80 i7p5eji _74wkf81">
                                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuidp eihuidn eihuid4z i7p5ej21" aria-hidden="true">
                                                <circle cx="12" cy="4" r="2"></circle>
                                                <circle cx="12" cy="20" r="2"></circle>
                                                <circle cx="12" cy="12" r="2"></circle>
                                              </svg>
                                            </span>
                                          </button>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                          <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6n eihuidhn _5qpagp0"></div>
                          <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                            <h1 class="gepq850 eihuid4z i7p5ej0 i7p5ejl _18ybopc4 i7p5ejs i7p5ej21" data-automation="job-detail-title">Mid-Level Frontend Developer</h1>
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgj eihuid6v eihuid6s eihuidhn _5qpagp0">
                              <button class="gepq850 gepq857 eihuidh _1h7s3kj0">
                                <span class="gepq850 eihuid4z eihuidi7 i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja" data-automation="advertiser-name">Complete Development (CoDev)
                                  <!-- -->
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                      <path d="m20.4 4.1-8-3c-.2-.1-.5-.1-.7 0l-8 3c-.4.1-.7.5-.7.9v7c0 6.5 8.2 10.7 8.6 10.9.1.1.2.1.4.1s.3 0 .4-.1c.4-.2 8.6-4.4 8.6-10.9V5c0-.4-.3-.8-.6-.9zM19 12c0 4.5-5.4 7.9-7 8.9-1.6-.9-7-4.3-7-8.9V5.7l7-2.6 7 2.6V12z"></path>
                                      <path d="M9.7 11.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l2 2c.2.2.5.3.7.3s.5-.1.7-.3l4-4c.4-.4.4-1 0-1.4s-1-.4-1.4 0L11 12.6l-1.3-1.3z"></path>
                                    </svg>
                                  </span>
                                </span>
                              </button>
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgn eihuid6n eihuidhn _5qpagp0">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7" data-automation="company-review">
                                  <span class="gepq850 ao7yqb0" aria-label="4.2 out of 5">
                                    <span class="gepq850 eihuid57">
                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej1z eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                        <path d="M23 9c-.1-.4-.4-.6-.8-.7l-6.4-.9-2.9-5.8c-.3-.5-.9-.7-1.4-.4-.2.1-.3.2-.4.4L8.2 7.3l-6.3 1c-.6.1-1 .6-.9 1.1 0 .2.1.4.3.6l4.6 4.5-1.1 6.4c-.1.5.3 1.1.8 1.2.2 0 .4 0 .6-.1l5.7-3 5.7 3c.5.3 1.1.1 1.3-.4.1-.2.1-.4.1-.6l-1.1-6.4 4.6-4.5c.5-.4.6-.8.5-1.1z"></path>
                                      </svg>
                                    </span>
                                  </span>
                                  <span class="gepq850 eihuidaz" aria-hidden="true">4.2</span>
                                </span>
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <a href="/companies/codev-168553861426412/reviews?jobId=82411354" rel="nofollow" class="gepq850 gepq85f  im98kj2 im98kj3 gepq850 gepq85f eihuidh" data-automation="job-header-company-review-link" target="_self">16 reviews</a>
                                </span>
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">·</span>
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <a href="/CoDev-jobs/at-this-company" rel="nofollow" class="gepq850 gepq85f  im98kj2 im98kj3 gepq850 gepq85f eihuidh" data-automation="job-details-header-more-jobs" target="_self">View all jobs</a>
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                          <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2d">
                              <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb7">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej22 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                      <path d="M12 1C7.6 1 4 4.6 4 9c0 4.1 6.5 12.6 7.2 13.6.2.2.5.4.8.4s.6-.1.8-.4c.7-1 7.2-9.5 7.2-13.6 0-4.4-3.6-8-8-8zm0 19.3c-2.2-3-6-8.8-6-11.3 0-3.3 2.7-6 6-6s6 2.7 6 6c0 2.5-3.8 8.3-6 11.3z"></path>
                                      <path d="M12 5c-1.7 0-3 1.3-3 3s1.3 3 3 3 3-1.3 3-3-1.3-3-3-3zm0 4c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1z"></path>
                                    </svg>
                                  </span>
                                </span>
                              </div>
                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidb7">
                                <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7" data-automation="job-detail-location">
                                    <a href="/Front-End-Developer-jobs/in-Manila-City-Metro-Manila" class="gepq850 gepq85f  gepq850 gepq85f _1ilznw00" tabindex="-1" target="_self">Manila City, Metro Manila</a>
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2d">
                              <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb7">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej22 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                      <path d="M9 6h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2z"></path>
                                      <path d="M17 2.2V2c0-.6-.4-1-1-1H8c-.6 0-1 .4-1 1v.2C5.9 2.6 5 3.7 5 5v16c0 .6.4 1 1 1h12c.6 0 1-.4 1-1V5c0-1.3-.9-2.4-2-2.8zM17 20h-3v-2h-4v2H7V5c0-.6.4-1 1-1h8c.6 0 1 .4 1 1v15z"></path>
                                    </svg>
                                  </span>
                                </span>
                              </div>
                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidb7">
                                <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7" data-automation="job-detail-classifications">
                                    <a href="/jobs-in-information-communication-technology/developers-programmers" class="gepq850 gepq85f  gepq850 gepq85f _1ilznw00" tabindex="-1" target="_self">Developers/Programmers (Information &amp; Communication Technology)</a>
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2d">
                              <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb7">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej22 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                      <path d="M16.4 13.1 13 11.4V6c0-.6-.4-1-1-1s-1 .4-1 1v6c0 .4.2.7.6.9l4 2c.1.1.2.1.4.1.4 0 .7-.2.9-.6.2-.4 0-1-.5-1.3z"></path>
                                      <path d="M12 1C5.9 1 1 5.9 1 12s4.9 11 11 11 11-4.9 11-11S18.1 1 12 1zm0 20c-5 0-9-4-9-9s4-9 9-9 9 4 9 9-4 9-9 9z"></path>
                                    </svg>
                                  </span>
                                </span>
                              </div>
                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidb7">
                                <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7" data-automation="job-detail-work-type">
                                    <a href="/Front-End-Developer-jobs/full-time" class="gepq850 gepq85f  gepq850 gepq85f _1ilznw00" tabindex="-1" target="_self">Full time</a>
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2d">
                              <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidb7">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej22 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                      <path fill="none" d="M19 7H5a2 2 0 0 1-2 2v6a2 2 0 0 1 2 2h14a2 2 0 0 1 2-2V9a2 2 0 0 1-2-2Z"></path>
                                      <path d="M22.945 7.45a3 3 0 0 0-2.396-2.395A2.988 2.988 0 0 0 20 5H4c-.188 0-.37.022-.55.055a3 3 0 0 0-2.395 2.396A2.988 2.988 0 0 0 1 8v8c0 .188.022.37.055.55a3 3 0 0 0 2.396 2.395c.178.033.361.055.549.055h16c.188 0 .37-.022.55-.055a3 3 0 0 0 2.395-2.396c.033-.178.055-.36.055-.549V8c0-.188-.022-.37-.055-.55ZM21 15a2 2 0 0 0-2 2H5a2 2 0 0 0-2-2V9a2 2 0 0 0 2-2h14a2 2 0 0 0 2 2v6Z"></path>
                                      <path d="M12 16c-2.206 0-4-1.794-4-4s1.794-4 4-4 4 1.794 4 4-1.794 4-4 4Zm0-6c-1.103 0-2 .897-2 2s.897 2 2 2 2-.897 2-2-.897-2-2-2Z"></path>
                                    </svg>
                                  </span>
                                </span>
                              </div>
                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidb7">
                                <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7" data-automation="job-detail-salary">₱45,000 – ₱55,000 per month</span>
                                </div>
                              </div>
                            </div>
                          </div>
                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">Posted 3d ago</span>
                        </div>
                        <div class="gepq850 eihuid4v eihuid50">
                          <div class="gepq850">
                            <div class="gepq850 eihuid4v eihuid50">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgf eihuid6r eihuidhn _5qpagp0">
                                <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2b _16qi62m2n">
                                  <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz eihuidb8">
                                    <div class="gepq850 eihuidp _1o5jf420">
                                      <a href="/job/82411354/apply" class="gepq850 gepq85f  gepq850 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka6 i7p5ej19 i7p5ej1b eihuid17 eihuid18" data-automation="job-detail-apply" aria-label="Apply for Mid-Level Frontend Developer at Complete Development (CoDev)" target="_self">
                                        <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                        <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 i7p5ej19 i7p5ej1b eihuid1b eihuid1c"></span>
                                        <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 i7p5ej19 i7p5ej1b eihuid19 eihuid1a"></span>
                                        <span class="gepq850 eihuidav eihuid9r eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                          <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">Quick apply</span>
                                        </span>
                                      </a>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidaz eihuidb8">
                                    <button class="gepq850 gepq857 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygkah vqygka0 vqygka6 i7p5ej18 i7p5ej1b eihuid27" type="button" aria-expanded="false" aria-label="Save Mid-Level Frontend Developer at Complete Development (CoDev)" data-testid="jdv-savedjob">
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1b eihuid2b eihuid2m"></span>
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1b eihuid29 eihuid2k"></span>
                                      <span class="gepq850 eihuidav eihuid9r eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                        <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5eja">
                                          <div class="gepq850 eihuid4z eihuid4w">
                                            <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                <path d="M19 5.1c.1-1.6-1.1-2.9-2.7-3.1H7.6C6.1 2.1 4.9 3.5 5 5v16c0 .4.2.7.5.9.3.2.7.2 1 0l5.4-3.6 5.4 3.6c.2.1.4.2.6.2.2 0 .3 0 .5-.1.3-.2.5-.5.5-.9l.1-16zm-2 14-4.4-3c-.3-.2-.8-.2-1.1 0l-4.4 3L7 4.9c0-.4.3-.9.7-.9h8.5c.5 0 .8.5.8 1v14.1z"></path>
                                              </svg>
                                            </span>
                                          </div>
                                          <div class="gepq850 eihuid4v eihuid50">Save</div>
                                        </span>
                                      </span>
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="gepq850 eihuid5b eihuidhf eihuid77">
                        <section>
                          <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                            <div data-automation="jobAdDetails">
                              <div class="gepq850 _1iptfqa0">
                                <p>The Front-End Developer will work on the <strong>healthcare price benchmarking tool</strong>, a web-based application that allows users to input metropolitan areas and healthcare billing codes (CPT codes) to receive relevant benchmark pricing data. This tool will include: </p>
                                <ul>
                                  <li>A <strong>searchable interface</strong> for users to input and retrieve pricing data </li>
                                  <li>
                                    <strong>Data visualization elements</strong> such as percentile ranges, regional heat maps, and interactive graphs
                                  </li>
                                  <li>
                                    <strong>Export functionality</strong> for reports and datasets
                                  </li>
                                </ul>
                                <p>The back-end, including data scrubbing and storage, will be handled by Simple Healthcare, with the front-end focused on presenting and visualizing this information.</p>
                                <p>This is the first step in a broader roadmap, with a more <strong>complex BI (Business Intelligence) dashboard</strong> planned for future development in 2026. </p>
                                <p>
                                  <strong>Responsibilities</strong>
                                </p>
                                <ul>
                                  <li>Develop a <strong>scalable, user-friendly front-end</strong> for the price benchmarking tool </li>
                                  <li>Implement <strong>search and filtering functionalities</strong> for healthcare pricing data </li>
                                  <li>Develop <strong>data visualization elements</strong> (heat maps, percentile charts, interactive tables) </li>
                                  <li>Optimize front-end performance for <strong>speed and accessibility</strong>
                                  </li>
                                  <li>Collaborate with a <strong>UI/UX designer</strong> to implement user-friendly designs </li>
                                  <li>Ensure the front-end integrates seamlessly with the back-end APIs</li>
                                  <li>Maintain <strong>clean, scalable, and well-documented</strong> code for future product expansion </li>
                                  <li>Work closely with stakeholders to gather and refine requirements</li>
                                </ul>
                                <p>
                                  <strong>Qualifications</strong>
                                </p>
                                <p>
                                  <strong>Must-Have:</strong>
                                </p>
                                <ul>
                                  <li>
                                    <strong>4+ years</strong> of front-end development experience
                                  </li>
                                  <li>Strong proficiency in <strong>React.js</strong> (preferred) or <strong>Vue.js</strong>
                                  </li>
                                  <li>Experience with <strong>Next.js</strong> (or similar SSR frameworks) </li>
                                  <li>Strong knowledge of <strong>JavaScript, TypeScript, HTML5, CSS3</strong>
                                  </li>
                                  <li>Experience with <strong>RESTful APIs and JSON data handling</strong>
                                  </li>
                                  <li>Familiarity with <strong>data visualization libraries</strong> (D3.js, Chart.js, Highcharts, or similar) </li>
                                  <li>Knowledge of <strong>state management</strong> (Redux, Context API) </li>
                                  <li>Understanding of <strong>responsive and accessible web design</strong>
                                  </li>
                                </ul>
                                <p>
                                  <strong>Nice-to-Have:</strong>
                                </p>
                                <ul>
                                  <li>Experience with <strong>BI tools</strong> or <strong>data-heavy applications</strong>
                                  </li>
                                  <li>Familiarity with <strong>Material UI, Tailwind CSS, or other UI libraries</strong>
                                  </li>
                                  <li>Experience with <strong>healthcare or financial tech applications</strong>
                                  </li>
                                  <li>Basic understanding of <strong>backend technologies (Node.js, Express.js, PostgreSQL/MongoDB)</strong>
                                  </li>
                                  <li>Exposure to <strong>React Native</strong> for potential future mobile development </li>
                                </ul>
                                <p>
                                  <strong>Additional Skill/ Ideal:</strong>
                                </p>
                                <ul>
                                  <li>Basic knowledge of <strong>SQL databases</strong>. </li>
                                  <li>Has understanding of <strong>UI/UX principles</strong> and front-end performance optimization. </li>
                                </ul>
                              </div>
                            </div>
                            <div class="gepq850 eihuid4z _17mt9yf0">
                              <div class="e4k0qg0">
                                <iframe src="https://www.youtube.com/embed/NFJmx-M_M4U?t=70s" class="e4k0qg1" allowfullscreen=""></iframe>
                              </div>
                            </div>
                          </div>
                        </section>
                        <section>
                          <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                            <h2 class="gepq850 eihuid4z i7p5ej0 i7p5ejl _18ybopc4 i7p5ejs i7p5ej21">Employer questions</h2>
                            <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">Your application will include the following questions:</span>
                              <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6r eihuidi7">
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">What\'s your expected monthly basic salary?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">Which of the following types of qualifications do you have?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">How many years\' experience do you have as a frontend software developer?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">Which of the following programming languages are you experienced in?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">How many years of front end development experience do you have?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">Which of the following front end development libraries and frameworks are you proficient in?</span>
                                  </div>
                                </li>
                                <li class="gepq850 eihuid5b">
                                  <div class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                    <div class="gepq850 eihuid5b eihuidgj eihuid4 if74j02" aria-hidden="true">
                                      <div class="gepq850 eihuid5v _16bj8lx0 _16bj8lx2"></div>
                                    </div>
                                  </div>
                                  <div class="gepq850 eihuidr eihuidp eihuidb7">
                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">How would you rate your English language skills?</span>
                                  </div>
                                </li>
                              </ul>
                            </div>
                          </div>
                        </section>
                        <section class="gepq850 gepq851">
                          <div class="gepq850">
                            <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                              <h3 class="gepq850 eihuid4z i7p5ej0 i7p5ejl _18ybopc4 i7p5ejs i7p5ej21">Company profile</h3>
                              <div class="gepq850 eihuid97 eihuid83 eihuidbf eihuidab eihuid6b eihuid4j eihuid4c" data-automation="company-profile">
                                <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                                  <section class="gepq850 gepq851">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                                      <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                                        <div class="gepq850 eihuid5b eihuidn">
                                          <img class="gepq850" src="https://image-service-cdn.seek.com.au/fce76906ed15a97d1c1f92c5b6b3125153f11c0e/ee4dce1061f3f616224767ad58cb2fc751b8d2dc" alt="Company Logo for CoDev" style="max-height:50px">
                                        </div>
                                        <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                          <button class="gepq850 gepq857 eihuidh _1r3rh851">
                                            <h4 class="gepq850 eihuid4z eihuidi7 i7p5ej0 i7p5ejl _18ybopc4 i7p5ejv i7p5ej21">CoDev
                                              <!-- -->
                                              <span class="gepq850 eihuid57">
                                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                  <path d="m20.4 4.1-8-3c-.2-.1-.5-.1-.7 0l-8 3c-.4.1-.7.5-.7.9v7c0 6.5 8.2 10.7 8.6 10.9.1.1.2.1.4.1s.3 0 .4-.1c.4-.2 8.6-4.4 8.6-10.9V5c0-.4-.3-.8-.6-.9zM19 12c0 4.5-5.4 7.9-7 8.9-1.6-.9-7-4.3-7-8.9V5.7l7-2.6 7 2.6V12z"></path>
                                                  <path d="M9.7 11.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l2 2c.2.2.5.3.7.3s.5-.1.7-.3l4-4c.4-.4.4-1 0-1.4s-1-.4-1.4 0L11 12.6l-1.3-1.3z"></path>
                                                </svg>
                                              </span>
                                            </h4>
                                          </button>
                                          <div class="gepq850 eihuid5b eihuidh7 eihuidgn eihuid6j eihuidhn _5qpagp0" data-automation="company-profile-review">
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja" data-automation="company-profile-review-rating">
                                              <span class="gepq850 ao7yqb0" aria-label="4.2 out of 5">
                                                <span class="gepq850 eihuid57">
                                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 i7p5ej1z eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                    <path d="M23 9c-.1-.4-.4-.6-.8-.7l-6.4-.9-2.9-5.8c-.3-.5-.9-.7-1.4-.4-.2.1-.3.2-.4.4L8.2 7.3l-6.3 1c-.6.1-1 .6-.9 1.1 0 .2.1.4.3.6l4.6 4.5-1.1 6.4c-.1.5.3 1.1.8 1.2.2 0 .4 0 .6-.1l5.7-3 5.7 3c.5.3 1.1.1 1.3-.4.1-.2.1-.4.1-.6l-1.1-6.4 4.6-4.5c.5-.4.6-.8.5-1.1z"></path>
                                                  </svg>
                                                </span>
                                              </span>
                                              <span class="gepq850 eihuidaz" aria-hidden="true">4.2</span>
                                            </span>
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">·</span>
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                              <a href="/companies/codev-168553861426412/reviews?jobId=82411354" rel="nofollow" class="gepq850 gepq85f  im98kj2 im98kj3 gepq850 gepq85f eihuidh" data-automation="company-profile-review-link" target="_self">16 reviews</a>
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </section>
                                  <section class="gepq850 gepq851">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                        <span class="gepq850 eihuid5b">
                                          <span class="gepq850 eihuid4z eihuid9v eihuidhz eihuidhv eihuidr">
                                            <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                <path d="M9 6h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9zm4 0h2v2h-2z"></path>
                                                <path d="M17 2.2V2c0-.6-.4-1-1-1H8c-.6 0-1 .4-1 1v.2C5.9 2.6 5 3.7 5 5v16c0 .6.4 1 1 1h12c.6 0 1-.4 1-1V5c0-1.3-.9-2.4-2-2.8zM17 20h-3v-2h-4v2H7V5c0-.6.4-1 1-1h8c.6 0 1 .4 1 1v15z"></path>
                                              </svg>
                                            </span>
                                          </span>
                                          <span class="gepq850 eihuid4z eihuidr">Information &amp; Communication Technology</span>
                                        </span>
                                      </span>
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                        <span class="gepq850 eihuid5b">
                                          <span class="gepq850 eihuid4z eihuid9v eihuidhz eihuidhv eihuidr">
                                            <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                <path d="M14.772 4.023c.076-.006.15-.023.228-.023a3 3 0 0 1 0 6c-.078 0-.152-.017-.228-.023a6.529 6.529 0 0 1-1.325 1.751A4.934 4.934 0 0 0 15 12a5 5 0 0 0 0-10 4.934 4.934 0 0 0-1.553.272 6.529 6.529 0 0 1 1.325 1.751ZM17 14h-1.853a6.54 6.54 0 0 1 1.613 2H17a3.003 3.003 0 0 1 3 3v2a1 1 0 0 0 2 0v-2a5.006 5.006 0 0 0-5-5Z"></path>
                                                <path d="M9 12a5 5 0 1 1 5-5 5.006 5.006 0 0 1-5 5Zm0-8a3 3 0 1 0 3 3 3.003 3.003 0 0 0-3-3Zm6 18a1 1 0 0 1-1-1v-2a3.003 3.003 0 0 0-3-3H7a3.003 3.003 0 0 0-3 3v2a1 1 0 0 1-2 0v-2a5.006 5.006 0 0 1 5-5h4a5.006 5.006 0 0 1 5 5v2a1 1 0 0 1-1 1Z"></path>
                                              </svg>
                                            </span>
                                          </span>
                                          <span class="gepq850 eihuid4z eihuidr">101-1,000 employees</span>
                                        </span>
                                      </span>
                                    </div>
                                  </section>
                                  <section class="gepq850 gepq851">
                                    <div class="gepq850">
                                      <div class="gepq850 eihuid5f">
                                        <div class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid0 eihuid6">
                                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                            <p class="gepq850 eihuidcz">Founded 14 years ago, CoDev connects highly-skilled developers from emerging nations such as the Philippines with small and medium-sized businesses in the US and Canada.</p>
                                            <p class="gepq850 eihuidcb">Our goal is to empower coders from the world’s most exciting markets, so they can partner with US and Canadian visionaries to create groundbreaking products for the online world.</p>
                                            <p class="gepq850 eihuidcb">Why Join CoDev:</p>
                                            <p class="gepq850 eihuidcb">Competitive Salary &amp; Benefits</p>
                                            <p class="gepq850 eihuidcb">Fun Culture &amp; Working Environment</p>
                                            <p class="gepq850 eihuidcb">Upskilling and Personal Growth</p>
                                            <p class="gepq850 eihuidcb">Work with U.S. Counterparts</p>
                                            <p class="gepq850 eihuidcb">Our Core Values:</p>
                                            <p class="gepq850 eihuidcb">Trustworthy</p>
                                            <p class="gepq850 eihuidcb">Reliable</p>
                                            <p class="gepq850 eihuidcb">Driven</p>
                                            <p class="gepq850 eihuidcb">Positive</p>
                                            <p class="gepq850 eihuidcb">Kind</p>
                                          </span>
                                        </div>
                                        <div class="gepq850" aria-hidden="true">
                                          <span class="gepq850 eihuid4z eihuidr i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja _1q03wcw0">
                                            <span class="gepq850 _1q03wcw2 eihuid4z eihuid0 eihuidr _1q03wcw4" style="--_1q03wcw3:3">
                                              <p class="gepq850 eihuidcz">Founded 14 years ago, CoDev connects highly-skilled developers from emerging nations such as the Philippines with small and medium-sized businesses in the US and Canada.</p>
                                              <p class="gepq850 eihuidcb">Our goal is to empower coders from the world’s most exciting markets, so they can partner with US and Canadian visionaries to create groundbreaking products for the online world.</p>
                                              <p class="gepq850 eihuidcb">Why Join CoDev:</p>
                                              <p class="gepq850 eihuidcb">Competitive Salary &amp; Benefits</p>
                                              <p class="gepq850 eihuidcb">Fun Culture &amp; Working Environment</p>
                                              <p class="gepq850 eihuidcb">Upskilling and Personal Growth</p>
                                              <p class="gepq850 eihuidcb">Work with U.S. Counterparts</p>
                                              <p class="gepq850 eihuidcb">Our Core Values:</p>
                                              <p class="gepq850 eihuidcb">Trustworthy</p>
                                              <p class="gepq850 eihuidcb">Reliable</p>
                                              <p class="gepq850 eihuidcb">Driven</p>
                                              <p class="gepq850 eihuidcb">Positive</p>
                                              <p class="gepq850 eihuidcb">Kind</p>
                                            </span>
                                          </span>
                                        </div>
                                      </div>
                                    </div>
                                  </section>
                                </div>
                                <div class="gepq850 eihuid87">
                                  <section class="gepq850 gepq851">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej3 i7p5ej21 _18ybopc4 i7p5eja">
                                        <span class="gepq850 eihuid5b">
                                          <span class="gepq850 eihuid4z eihuid9v eihuidhz eihuidhv eihuidr">
                                            <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                <path d="M21 6h-3.184c.112-.314.184-.648.184-1 0-1.654-1.346-3-3-3-1.395 0-2.363 1.101-3 2.259C11.363 3.1 10.395 2 9 2 7.346 2 6 3.346 6 5c0 .352.072.686.184 1H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h1v7c0 1.654 1.346 3 3 3h10c1.654 0 3-1.346 3-3v-7h1a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1Zm-6-2a1 1 0 0 1 0 2h-1.61c.435-1.034 1.056-2 1.61-2ZM8 5a1 1 0 0 1 1-1c.554 0 1.175.966 1.61 2H9a1 1 0 0 1-1-1ZM4 8h7v2H4V8Zm2 11v-7h5v8H7a1 1 0 0 1-1-1Zm12 0a1 1 0 0 1-1 1h-4v-8h5v7Zm2-9h-7V8h7v2Z"></path>
                                              </svg>
                                            </span>
                                          </span>
                                          <span class="gepq850 eihuid4z eihuidr">Perks and benefits</span>
                                        </span>
                                      </span>
                                      <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgg eihuid6v eihuidhn _5qpagp1 _5qpagp2 _5qpagp3">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Life Insurance, Health/HMO Insurance +2 Dependents</div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Miscellaneous allowance</div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Annual Salary Increase </div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Remote Work Flexibility</div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Company Employee Morale Activities</div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Upskill / Online Training</div>
                                        </span>
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                          <div class="gepq850 _1r3rh850">Company Issued Machine</div>
                                        </span>
                                      </div>
                                    </div>
                                  </section>
                                </div>
                                <div class="gepq850 eihuid87">
                                  <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgg eihuid6n eihuidhn _5qpagp1 _5qpagp2 _5qpagp3">
                                    <a href="/companies/codev-168553861426412" class="gepq850 gepq85f  gepq850 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka6" data-automation="company-profile-profile-link" target="_self">
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1a eihuid2t eihuid2u"></span>
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1a eihuid2r eihuid2s"></span>
                                      <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid63 eihuid5j eihuidi eihuid4b eihuid4g"></span>
                                      <span class="gepq850 eihuidav eihuid9r eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9 i7p5ej18 i7p5ej1b ">
                                        <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">More about this company <span class="gepq850 eihuidaz smf27g0 _16qi62m3f" aria-hidden="true">\u2060 <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 _1fzo8jp0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84 _1fzo8jp3" aria-hidden="true">
                                                <path d="m11.293 4.293-7 7a1 1 0 1 0 1.414 1.414L11 7.414V19a1 1 0 1 0 2 0V7.414l5.293 5.293a1 1 0 1 0 1.414-1.414l-7-7a1 1 0 0 0-1.414 0Z"></path>
                                              </svg>
                                            </span>
                                          </span>
                                        </span>
                                      </span>
                                    </a>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </section>
                        <section>
                          <div class="gepq850">
                            <div class="gepq850 eihuid5j eihuid0 _1qnq8v60">
                              <h2>Report this job advert</h2>
                            </div>
                            <div class="gepq850 eihuid5b eihuidhf eihuid6v">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5ejd">Be careful</span>
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">Don’t provide your bank or credit card details when applying for jobs.</span>
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                <a href="/security-privacy" rel="nofollow" class="gepq850 gepq85f  im98kj2 im98kj3 gepq850 gepq85f eihuidh f9bd8p0" target="_self">Learn how to protect yourself</a>
                              </span>
                              <div class="gepq850 eihuid4z _17mt9yf0">
                                <div class="gepq850 eihuidcb">
                                  <div class="gepq850" data-automation="report-job-ad-toggle">
                                    <div class="gepq850 eihuid4">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                        <span class="gepq850 im98kj2 im98kj4 im98kj7 i7p5ej2 eihuidh f9bd8p0" role="button" tabindex="0" aria-controls="toggleReportJobAdForm" aria-expanded="false">Report this job ad <span class="gepq850 eihuidaz smf27g0" aria-hidden="true">\u2060 <span class="gepq850 eihuid57">
                                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                                <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                              </svg>
                                            </span>
                                          </span>
                                        </span>
                                      </span>
                                    </div>
                                    <div class="gepq850 eihuid83 eihuid4v" id="toggleReportJobAdForm">
                                      <form data-automation="report-job-ad-form">
                                        <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                                          <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                            <div class="gepq850 eihuid5b eihuidhf eihuid6r eihuidi7">
                                              <span class="gepq850 eihuid5b eihuidh3">
                                                <label for="email">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                                    <strong class="i7p5ej3">Your email address</strong>
                                                  </span>
                                                </label>
                                              </span>
                                            </div>
                                            <div class="gepq850 eihuid5b eihuidhf eihuid6n">
                                              <div class="gepq850 eihuid5f eihuid63 eihuid5b i7p5ej18  eihuid33">
                                                <input class="gepq850 gepq851 gepq856 gepq857 gepq85c eihuidp eihuidb7 eihuida3 eihuid63 eihuid5 rvw75g0 rvw75g1 i7p5ej0 i7p5ej1 i7p5ej22 i7p5eji i7p5ej26 i7p5ej18  eihuid33" type="text" id="email" data-automation="report-job-ad-email" inputmode="text" value="">
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid3x eihuid3y"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 eihuid3p eihuid3u"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g5 eihuid4t eihuid4u"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g6 eihuid3z eihuid44"></span>
                                              </div>
                                            </div>
                                          </div>
                                          <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                            <div class="gepq850 eihuid5b eihuidhf eihuid6r eihuidi7">
                                              <span class="gepq850 eihuid5b eihuidh3">
                                                <label for="reason">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                                    <strong class="i7p5ej3">Reason for reporting job</strong>
                                                  </span>
                                                </label>
                                              </span>
                                            </div>
                                            <div class="gepq850 eihuid5b eihuidhf eihuid6n">
                                              <div class="gepq850 eihuid5f eihuid63 eihuid5b i7p5ej18  eihuid33">
                                                <select class="gepq850 gepq851 gepq856 gepq857 gepq85a eihuidp eihuidb7 eihuid63 eihuid5 _1jz3g80 rvw75g0 rvw75g1 i7p5ej0 i7p5ej1 i7p5ej22 i7p5eji i7p5ej26 i7p5ej18  eihuid33" placeholder="Please select" id="reason" data-automation="report-job-ad-reason">
                                                  <option value="" disabled="" selected="">Please select</option>
                                                  <option value="Fraudulent">Fraudulent</option>
                                                  <option value="Discrimination">Discrimination</option>
                                                  <option value="Misleading">Misleading</option>
                                                </select>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid3x eihuid3y"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 eihuid3p eihuid3u"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g5 eihuid4t eihuid4u"></span>
                                                <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g6 eihuid3z eihuid44"></span>
                                                <div class="gepq850 eihuid5j eihuid5b eihuidgj eihuidgv eihuidi eihuido eihuidq eihuidj eihuidm">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 i7p5eji">
                                                    <span class="gepq850 eihuid57">
                                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                        <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                                      </svg>
                                                    </span>
                                                  </span>
                                                </div>
                                              </div>
                                            </div>
                                          </div>
                                          <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                            <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                                              <div class="gepq850 eihuid5b eihuidhf eihuid6r eihuidi7">
                                                <span class="gepq850 eihuid5b eihuidh3">
                                                  <label for="comment">
                                                    <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                                      <strong class="i7p5ej3">Additional comments</strong>
                                                    </span>
                                                  </label>
                                                </span>
                                              </div>
                                              <div class="gepq850 eihuid5b eihuidhf eihuid6n">
                                                <div class="gepq850 eihuid5f eihuid63 eihuid5b i7p5ej18  eihuid33">
                                                  <div class="gepq850 eihuid5f eihuidp eihuid7 eihuid63 i7p5ej18  eihuid33">
                                                    <textarea class="gepq850 gepq851 gepq856 gepq857 eihuid5f eihuid8 eihuidp eihuidb7 eihuida3 eihuid5 ypxads0 rvw75g0 rvw75g1 i7p5ej0 i7p5ej1 i7p5ej22 i7p5eji i7p5ej26" rows="3" id="comment" data-automation="report-job-ad-comment"></textarea>
                                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid3x eihuid3y"></span>
                                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 eihuid3p eihuid3u"></span>
                                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g5 eihuid4t eihuid4u"></span>
                                                    <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 rvw75g6 eihuid3z eihuid44"></span>
                                                  </div>
                                                </div>
                                              </div>
                                            </div>
                                            <div class="gepq850 eihuid5b eihuidgz eihuidi7" id="comment-message">
                                              <div class="gepq850 eihuidi3">
                                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej1v _18ybopc4 i7p5ej7">
                                                  <span class="gepq850 eihuid5b">
                                                    <span class="gepq850 eihuid4z eihuid9v eihuidhz eihuidhv eihuidr">
                                                      <span class="gepq850 eihuid57">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 i7p5ej1v eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                          <path d="M22.71 17.262 14.738 3.71A3.183 3.183 0 0 0 12 2.013 3.183 3.183 0 0 0 9.262 3.71L1.29 17.262a3.152 3.152 0 0 0-.14 3.225A3.152 3.152 0 0 0 4 22h16a3.023 3.023 0 0 0 2.71-4.738ZM20 20H4c-1.1 0-1.544-.776-.986-1.724l7.972-13.552A1.232 1.232 0 0 1 12 4.013a1.232 1.232 0 0 1 1.014.71l7.972 13.553C21.544 19.224 21.1 20 20 20Z"></path>
                                                          <circle cx="12" cy="17" r="1"></circle>
                                                          <path d="M11.978 14a1 1 0 0 0 1-1V9a1 1 0 0 0-2 0v4a1 1 0 0 0 1 1Z"></path>
                                                        </svg>
                                                      </span>
                                                    </span>
                                                    <span class="gepq850 eihuid4z eihuidr">To help fast track investigation, please include here any other relevant details that prompted you to report this job ad as fraudulent / misleading / discriminatory.</span>
                                                  </span>
                                                </span>
                                              </div>
                                            </div>
                                          </div>
                                          <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgg eihuid6n eihuidhn _5qpagp1 _5qpagp2 _5qpagp3">
                                            <button class="gepq850 gepq857 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka6 i7p5ej19 i7p5ej1b eihuid21 eihuid22" type="submit" data-automation="report-job-ad-submit">
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 i7p5ej19 i7p5ej1b eihuid25 eihuid26"></span>
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 i7p5ej19 i7p5ej1b eihuid23 eihuid24"></span>
                                              <span class="gepq850 eihuidav eihuid9r eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                                <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">Report job</span>
                                              </span>
                                            </button>
                                            <button class="gepq850 gepq857 eihuid63 eihuidp eihuid5f eihuid5b eihuidgj eihuidgv eihuidz eihuidy eihuid5 eihuidib eihuid4 eihuidh vqygka0 vqygka6" type="button" data-automation="report-job-ad-cancel">
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 _1g7rtu60 vqygka4 eihuid4t eihuid4u"></span>
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka3 vqygkai i7p5ej18 i7p5ej1b eihuid2b eihuid2m"></span>
                                              <span class="gepq850 eihuidj eihuidk eihuidl eihuidm eihuid5j eihuidi eihuid63 eihuidx eihuid6 vqygka2 vqygkaj i7p5ej18 i7p5ej1b eihuid29 eihuid2k"></span>
                                              <span class="gepq850 eihuidb7 eihuida3 eihuid5f eihuid5b eihuidgv eihuidi3 eihuidhn eihuid0 eihuidi vqygka9">
                                                <span class="gepq850 eihuid4z eihuidib i7p5ej0 i7p5ej2 i7p5ej20 _18ybopc4 i7p5eja">Cancel</span>
                                              </span>
                                            </button>
                                          </div>
                                        </div>
                                      </form>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </section>
                        <section>
                          <div class="gepq850 eihuid8n eihuid8o eihuid9d eihuid7j eihuid7k eihuid89 eihuidav eihuidaw eihuidbl eihuid9r eihuid9s eihuidah eihuid67 eihuid4j eihuid4c" style="margin:auto">
                            <div class="gepq850" style="max-width:100%;margin:auto">
                              <div>
                                <div data-automation="dynamic-lmis" style="z-index:0;position:relative">
                                  <style>
                                    \n

                                    /* capsize font, don\'t change */
                                    \n .capsize-heading4 {
                                      \n font-size: 20px;
                                      \n line-height: 24.66px;
                                      \n display: block;
                                      \n font-weight: 500;
                                      \n
                                    }

                                    \n\n .capsize-heading4::before {
                                      \n content: \'\';
                                      \n margin-bottom: -0.225em;
                                      \n display: table;
                                      \n
                                    }

                                    \n\n .capsize-heading4::after {
                                      \n content: \'\';
                                      \n margin-top: -0.225em;
                                      \n display: table;
                                      \n
                                    }

                                    \n\n .capsize-standardText {
                                      \n font-size: 16px;
                                      \n line-height: 24.528px;
                                      \n display: block;
                                      \n
                                    }

                                    \n\n .capsize-standardText::before {
                                      \n content: \'\';
                                      \n margin-bottom: -0.375em;
                                      \n display: table;
                                      \n
                                    }

                                    \n\n .capsize-standardText::after {
                                      \n content: \'\';
                                      \n margin-top: -0.375em;
                                      \n display: table;
                                      \n
                                    }

                                    \n\n @media only screen and (min-width: 740px) {
                                      \n .capsize-heading3 {
                                        \n font-size: 24px;
                                        \n line-height: 29.792px;
                                        \n
                                      }

                                      \n\n .capsize-heading3::before {
                                        \n content: \'\';
                                        \n margin-bottom: -0.2292em;
                                        \n display: table;
                                        \n
                                      }

                                      \n\n .capsize-heading3::after {
                                        \n content: \'\';
                                        \n margin-top: -0.2292em;
                                        \n display: table;
                                        \n
                                      }

                                      \n
                                    }

                                    \n
                                    /* end of capsize */
                                    \n\n

                                    /* LMIS css start here*/
                                    \n .lmis-root {
                                      \n margin: -32px;
                                      \n padding: 32px;
                                      \n font-family: SeekSans, \'SeekSans Fallback\', Arial, sans-serif;
                                      \n background: #beeff3;
                                      \n border-radius: 16px;
                                      \n color: #2e3849;
                                      \n
                                    }

                                    \n\n .lmis-title {
                                      \n margin-bottom: 8px;
                                      \n
                                    }

                                    \n\n .lmis-cta {
                                      \n min-height: 48px;
                                      \n display: flex;
                                      \n align-items: center;
                                      \n color: #2e3849;
                                      \n text-decoration: none;
                                      \n
                                    }

                                    \n\n .lmis-cta-text {
                                      \n margin-right: 4px;
                                      \n font-weight: 500;
                                      \n
                                    }

                                    \n\n .lmis-teaser-image {
                                      \n max-width: 96px;
                                      \n
                                    }

                                    \n\n @media only screen and (min-width: 992px) {
                                      \n .lmis-root {
                                        \n margin: -48px;
                                        \n
                                      }

                                      \n\n .lmis-wrapper {
                                        \n display: flex;
                                        \n flex-direction: row-reverse;
                                        \n justify-content: space-between;
                                        \n align-items: center;
                                        \n
                                      }

                                      \n
                                    }

                                    \n
                                  </style>\n\n <div class="lmis-root">\n <div class="lmis-wrapper">\n <div class="lmis-teaser-image">\n <img src="https://cdn.seeklearning.com.au/media/images/lmis/girl_comparing_salaries.svg" alt="salary teaser image">\n </div>\n <div class="lmis-content">\n <div class="capsize-heading4 lmis-title">What can I earn as a Front End Developer</div>\n <a class="lmis-cta" href="https://ph.jobstreet.com/career-advice/role/frontend-developer/salary?campaigncode=lrn:skj:sklm:cg:jbd:alpha" target="_blank">\n <span class="capsize-standardText lmis-cta-text">See more detailed salary information</span>\n <img src="https://cdn.seeklearning.com.au/media/images/lmis/arrow_right.svg" alt="salary teaser link arrow">\n </a>\n </div>\n </div>\n </div>\n
                                </div>
                              </div>
                            </div>
                          </div>
                        </section>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="gepq850 eihuid8b">
          <footer class="gepq850 gepq851 eihuid5f i7p5ej18 i7p5ej1b eihuid11 eihuid14" data-automation="footer">
            <div class="gepq850 eihuidb3 eihuidaw eihuid9z eihuid9s">
              <div class="gepq850 eihuidp eihuidv gfmryc0">
                <div class="gepq850 eihuid87">
                  <div class="gepq850 eihuid5b eihuidhf eihuid73">
                    <div class="gepq850 eihuid5b eihuidhf eihuidh8 eihuidgr eihuidn _16qi62m2j _16qi62m2o">
                      <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbv eihuidbc eihuid7z eihuid8k _17256u60 _17256u61">
                        <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2e">
                          <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbb _17256u61">
                            <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">Job seekers</span>
                              <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                <li class="gepq850">
                                  <a href="/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="job search" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Job search</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/profile" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="profile" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Profile</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/oauth/login/?returnUrl=%2Frecommended" rel="nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="recommended jobs" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Recommended jobs</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/oauth/login/?returnUrl=%2Fmy-activity%2Fsaved-searches" rel="nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="saved searches" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Saved searches</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/oauth/login/?returnUrl=%2Fmy-activity%2Fsaved-jobs" rel="nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="saved jobs" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Saved jobs</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/oauth/login/?returnUrl=%2Fmy-activity%2Fapplied-jobs" rel="nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="applied jobs" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Job applications</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/career-advice" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="career advice" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Career advice</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/career-advice/explore-careers" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="explore careers" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Explore careers</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/career-advice/explore-salaries" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="explore salaries" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Explore salaries</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/companies" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="company reviews" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Explore companies</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="/community/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="community" target="_self">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Community</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <div class="gepq850 eihuid5b eihuidhf eihuid5f">
                                    <div class="gepq850 _19i2h2t0" data-automation="download apps" aria-expanded="false">
                                      <div class="gepq850 eihuideb">
                                        <div class="gepq850 s7ult00">
                                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                            <button class="gepq850 gepq857 s7ult01">Download apps</button>
                                          </span>
                                        </div>
                                      </div>
                                      <div class="gepq850">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                            <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                              <span class="gepq850 eihuid57">
                                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                                  <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                                </svg>
                                              </span>
                                            </div>
                                          </div>
                                        </span>
                                      </div>
                                    </div>
                                    <div class="gepq850 _19i2h2t2">
                                      <div class="gepq850 eihuid83 eihuidb7">
                                        <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                          <li class="gepq850">
                                            <a href="https://play.google.com/store/apps/details?id=com.jobstreet.jobstreet&amp;hl=en" rel="noopener nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="android" target="_blank">
                                              <div class="gepq850 eihuid5b eihuidgj">
                                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                  <span class="gepq850 eihuidef">
                                                    <span class="gepq850 eihuid57">
                                                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                        <path d="m17.09 10.255 1.674-2.891a.577.577 0 0 0-.237-.773.58.58 0 0 0-.754.2l-1.71 2.945a10.42 10.42 0 0 0-8.127 0L6.227 6.791a.584.584 0 0 0-.79-.182.58.58 0 0 0-.2.755l1.672 2.89A9.8 9.8 0 0 0 2 18h20a9.8 9.8 0 0 0-4.91-7.745ZM7.456 15.5a1.137 1.137 0 1 1 0-2.274 1.137 1.137 0 0 1 0 2.274Zm9.09 0a1.137 1.137 0 1 1 .001-2.274 1.137 1.137 0 0 1 0 2.274Z"></path>
                                                      </svg>
                                                    </span>
                                                  </span>
                                                  <span class="gepq850 _12f6icc1">Jobstreet @ Google Play</span>
                                                </span>
                                              </div>
                                            </a>
                                          </li>
                                          <li class="gepq850">
                                            <a href="https://apps.apple.com/ph/app/jobstreet/id367294492" rel="noopener nofollow" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="ios" target="_blank">
                                              <div class="gepq850 eihuid5b eihuidgj">
                                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                  <span class="gepq850 eihuidef">
                                                    <span class="gepq850 eihuid57">
                                                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" focusable="false" fill="currentColor" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                        <path d="M17.957 8.241c-.104.081-1.948 1.12-1.948 3.43 0 2.67 2.346 3.616 2.416 3.64-.01.057-.373 1.294-1.237 2.554-.77 1.109-1.575 2.216-2.799 2.216s-1.54-.711-2.952-.711c-1.377 0-1.867.734-2.987.734s-1.9-1.026-2.799-2.286c-1.04-1.48-1.881-3.779-1.881-5.96 0-3.5 2.275-5.356 4.515-5.356 1.19 0 2.181.781 2.929.781.71 0 1.82-.828 3.173-.828.513 0 2.357.047 3.57 1.786Zm-4.212-3.267c.56-.665.955-1.586.955-2.508 0-.128-.01-.257-.034-.362-.91.034-1.994.607-2.648 1.365-.513.583-.992 1.505-.992 2.439 0 .14.024.28.035.326.057.01.15.023.244.023.818 0 1.846-.547 2.44-1.283Z"></path>
                                                      </svg>
                                                    </span>
                                                  </span>
                                                  <span class="gepq850 _12f6icc1">Jobstreet @ App Store</span>
                                                </span>
                                              </div>
                                            </a>
                                          </li>
                                        </ul>
                                      </div>
                                    </div>
                                  </div>
                                </li>
                              </ul>
                            </div>
                          </div>
                          <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbb _17256u61">
                            <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">Employers</span>
                              <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="register for free" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Register for free</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="post a job ad" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Post a job ad</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/products/jobads" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="products &amp; prices" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Products &amp; prices</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/contactus" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="customer service" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Customer service</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/hiring-advice" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="hiring Advice" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Hiring advice</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/market-insights" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="market insights" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Market insights</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                                <li class="gepq850">
                                  <a href="https://ph.employer.seek.com/partners/connect-with-seek" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="recruitment software partners" target="_blank">
                                    <div class="gepq850 eihuid5b eihuidgj">
                                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                        <span class="gepq850 _12f6icc1">Recruitment software partners</span>
                                      </span>
                                    </div>
                                  </a>
                                </li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbv eihuidbc eihuid7z eihuid8k _17256u60 _17256u61">
                        <div class="gepq850 eihuidcb eihuidd0">
                          <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidn _16qi62m2e">
                            <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbb _17256u61">
                              <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">About us</span>
                                <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                  <li class="gepq850">
                                    <a href="/about" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="about seek" target="_self">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">About us</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <a href="/about/news" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="newsroom" target="_self">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Newsroom</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <a href="https://www.seek.com.au/about/investors" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="investors" target="_self">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Investors</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <a href="/careers" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="work for seek" target="_blank">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Careers</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid5f">
                                      <div class="gepq850 _19i2h2t0" data-automation="international partners" aria-expanded="false">
                                        <div class="gepq850 eihuideb">
                                          <div class="gepq850 s7ult00">
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                              <button class="gepq850 gepq857 s7ult01">International partners</button>
                                            </span>
                                          </div>
                                        </div>
                                        <div class="gepq850">
                                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                                <span class="gepq850 eihuid57">
                                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                                    <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                                  </svg>
                                                </span>
                                              </div>
                                            </div>
                                          </span>
                                        </div>
                                      </div>
                                      <div class="gepq850 _19i2h2t2">
                                        <div class="gepq850 eihuid83 eihuidb7">
                                          <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                            <li class="gepq850">
                                              <a href="https://www.bdjobs.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="bdjobs" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">Bdjobs (Bangladesh)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.jobsdb.com/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="jobsdb" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">Jobsdb (SE Asia)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://au.jora.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="jora au" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">Jora (Australia)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://jora.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="jora global" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">Jora (Worldwide)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.seek.com.au/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="seek" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">SEEK (Australia)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.seek.co.nz/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="seek" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">SEEK (New Zealand)</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                          </ul>
                                        </div>
                                      </div>
                                    </div>
                                  </li>
                                  <li class="gepq850">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid5f">
                                      <div class="gepq850 _19i2h2t0" data-automation="partner services" aria-expanded="false">
                                        <div class="gepq850 eihuideb">
                                          <div class="gepq850 s7ult00">
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                              <button class="gepq850 gepq857 s7ult01">Partner services</button>
                                            </span>
                                          </div>
                                        </div>
                                        <div class="gepq850">
                                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                                <span class="gepq850 eihuid57">
                                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                                    <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                                  </svg>
                                                </span>
                                              </div>
                                            </div>
                                          </span>
                                        </div>
                                      </div>
                                      <div class="gepq850 _19i2h2t2">
                                        <div class="gepq850 eihuid83 eihuidb7">
                                          <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                            <li class="gepq850">
                                              <a href="https://seekpass.co/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="seek pass" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">SEEK Pass</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://au.gradconnection.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="gradconnection" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">GradConnection</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://sidekicker.com/au/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="sidekicker" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">Sidekicker</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.go1.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="go1" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">GO1</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.futurelearn.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="futurelearn" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">FutureLearn</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://jobadder.com/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="jobadder" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 _12f6icc1">JobAdder</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                          </ul>
                                        </div>
                                      </div>
                                    </div>
                                  </li>
                                </ul>
                              </div>
                            </div>
                            <div class="gepq850 eihuid4z eihuidr eihuidp eihuidhv eihuidhz eihuidbb _17256u61">
                              <div class="gepq850 eihuid5b eihuidhf eihuid6z">
                                <span class="gepq850 eihuid4z i7p5ej0 i7p5ej2 i7p5ej21 _18ybopc4 i7p5eja">Contact</span>
                                <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                  <li class="gepq850">
                                    <a href="https://help.ph.jobstreet.com/s/" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="help centre" target="_blank">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Help centre</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <a href="/contact-us" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="contact us" target="_self">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Contact us</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <a href="https://medium.com/seek-blog" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="product &amp; tech blog" target="_blank">
                                      <div class="gepq850 eihuid5b eihuidgj">
                                        <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                          <span class="gepq850 _12f6icc1">Product &amp; tech blog</span>
                                        </span>
                                      </div>
                                    </a>
                                  </li>
                                  <li class="gepq850">
                                    <div class="gepq850 eihuid5b eihuidhf eihuid5f">
                                      <div class="gepq850 _19i2h2t0" data-automation="social" aria-expanded="false">
                                        <div class="gepq850 eihuideb">
                                          <div class="gepq850 s7ult00">
                                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                              <button class="gepq850 gepq857 s7ult01">Social</button>
                                            </span>
                                          </div>
                                        </div>
                                        <div class="gepq850">
                                          <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                            <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                              <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                                <span class="gepq850 eihuid57">
                                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                                    <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                                  </svg>
                                                </span>
                                              </div>
                                            </div>
                                          </span>
                                        </div>
                                      </div>
                                      <div class="gepq850 _19i2h2t2">
                                        <div class="gepq850 eihuid83 eihuidb7">
                                          <ul class="gepq850 gepq853 eihuid5b eihuidhf eihuid6z">
                                            <li class="gepq850">
                                              <a href="https://www.facebook.com/JobstreetPH/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="facebook" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 eihuidef">
                                                      <span class="gepq850 eihuid57">
                                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                          <path d="M15.016 2c.8 0 2.183.157 2.748.314v3.502a16.218 16.218 0 0 0-1.46-.047c-2.074 0-2.875.786-2.875 2.827v1.367h3.85l-.71 3.863h-3.14V22H8.843v-8.174H6.236V9.963h2.607V8.298C8.843 3.995 10.791 2 15.015 2"></path>
                                                        </svg>
                                                      </span>
                                                    </span>
                                                    <span class="gepq850 _12f6icc1">Facebook</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.instagram.com/jobstreetph/" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="instagram" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 eihuidef">
                                                      <span class="gepq850 eihuid57">
                                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                          <path d="M7.858 2.07c-1.064.05-1.79.22-2.425.469-.658.256-1.215.6-1.77 1.156a4.9 4.9 0 0 0-1.15 1.773c-.246.637-.413 1.364-.46 2.429-.046 1.066-.057 1.407-.052 4.122.005 2.715.017 3.056.068 4.123.051 1.064.22 1.79.47 2.426.256.657.6 1.214 1.156 1.769a4.904 4.904 0 0 0 1.774 1.15c.636.245 1.363.413 2.428.46 1.066.047 1.407.057 4.122.052 2.716-.005 3.056-.017 4.123-.068 1.064-.05 1.79-.221 2.426-.47a4.901 4.901 0 0 0 1.769-1.156 4.903 4.903 0 0 0 1.15-1.774c.245-.636.413-1.363.46-2.427.046-1.067.057-1.408.052-4.123-.005-2.716-.018-3.056-.068-4.122-.05-1.065-.221-1.79-.47-2.427a4.905 4.905 0 0 0-1.156-1.769 4.894 4.894 0 0 0-1.774-1.15c-.636-.245-1.363-.413-2.427-.46-1.067-.047-1.407-.057-4.123-.052-2.715.005-3.056.017-4.123.068m.117 18.078c-.975-.043-1.504-.205-1.857-.34a3.136 3.136 0 0 1-1.152-.746 3.107 3.107 0 0 1-.75-1.149c-.137-.352-.302-.881-.347-1.856-.05-1.054-.061-1.37-.066-4.04-.005-2.67.004-2.986.05-4.041.042-.974.205-1.504.34-1.857.182-.468.398-.8.747-1.151.35-.351.682-.568 1.148-.75.353-.138.881-.302 1.856-.348 1.055-.05 1.37-.06 4.04-.066 2.67-.005 2.986.004 4.041.05.975.043 1.505.204 1.857.34.467.182.8.397 1.151.747.35.35.568.681.75 1.149.138.351.302.88.348 1.855.05 1.054.062 1.37.066 4.04.006 2.67-.004 2.986-.05 4.04-.043.976-.205 1.505-.34 1.859-.181.467-.398.8-.747 1.15a3.11 3.11 0 0 1-1.148.75c-.352.138-.882.302-1.856.349-1.054.05-1.37.06-4.04.065-2.67.006-2.986-.005-4.04-.05m8.151-13.492a1.2 1.2 0 1 0 2.4-.006 1.2 1.2 0 0 0-2.4.006M6.865 12.01a5.134 5.134 0 1 0 10.269-.02 5.134 5.134 0 0 0-10.269.02m1.802-.004a3.333 3.333 0 1 1 6.666-.011 3.333 3.333 0 0 1-6.666.011"></path>
                                                        </svg>
                                                      </span>
                                                    </span>
                                                    <span class="gepq850 _12f6icc1">Instagram</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://twitter.com/JobStreetPH" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="twitter" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 eihuidef">
                                                      <span class="gepq850 eihuid57">
                                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                          <path d="M14.096 10.55 20.666 3h-2l-5.404 6.21-3.278-5.267L9.397 3H2.399l1.903 3.057 4.88 7.84L3 21h2l5.016-5.763 3 4.82.587.943h6.998l-1.903-3.057-4.602-7.393Zm.618 8.45-3.324-5.341-.834-1.34L6 5h2.286l3.602 5.788.834 1.34L17 19h-2.286Z"></path>
                                                        </svg>
                                                      </span>
                                                    </span>
                                                    <span class="gepq850 _12f6icc1">Twitter</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                            <li class="gepq850">
                                              <a href="https://www.youtube.com/@JobStreetPhilippines" rel="noopener" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="youtube" target="_blank">
                                                <div class="gepq850 eihuid5b eihuidgj">
                                                  <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej7">
                                                    <span class="gepq850 eihuidef">
                                                      <span class="gepq850 eihuid57">
                                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf84" aria-hidden="true">
                                                          <path d="M22.54 6.705a2.755 2.755 0 0 0-1.945-1.945C18.88 4.3 12 4.3 12 4.3s-6.88 0-8.595.46A2.755 2.755 0 0 0 1.46 6.705C1 8.42 1 12 1 12s0 3.58.46 5.295a2.755 2.755 0 0 0 1.945 1.945C5.12 19.7 12 19.7 12 19.7s6.88 0 8.595-.46a2.755 2.755 0 0 0 1.945-1.945C23 15.58 23 12 23 12s0-3.58-.46-5.295ZM9.8 15.3V8.7l5.716 3.3L9.8 15.3Z"></path>
                                                        </svg>
                                                      </span>
                                                    </span>
                                                    <span class="gepq850 _12f6icc1">YouTube</span>
                                                  </span>
                                                </div>
                                              </a>
                                            </li>
                                          </ul>
                                        </div>
                                      </div>
                                    </div>
                                  </li>
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <span class="gepq850 eihuid4z eihuid5f eihuidp">
                      <span class="gepq850 eihuid5j eihuidl eihuidm _16n2gzb0 _16n2gzb3 _16n2gzb5 _16n2gzb8"></span>
                    </span>
                  </div>
                </div>
                <div class="gepq850 eihuid83 eihuid9b">
                  <div class="gepq850 eihuid5b eihuidgj eihuidgg eihuidgq eihuidgr eihuidh4 eihuidh7 eihuidhg eihuidha eihuidhq">
                    <div class="gepq850 eihuid4v eihuid50 eihuid98 eihuid9q">
                      <div class="gepq850 eihuid5b eihuidhf eihuid6r">
                        <div class="gepq850 eihuid5b eihuidhf eihuid6r eihuidi7">
                          <span class="gepq850 eihuid5b eihuidh3">
                            <label id="country-label" for="country-switcher-menu">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5eja">
                                <strong class="i7p5ej3">Country</strong>
                              </span>
                            </label>
                          </span>
                        </div>
                        <div class="gepq850" data-automation="country-switcher-menu">
                          <button class="gepq850 gepq857 eihuid4 eihuidh" data-testid="country-switcher-menu" aria-haspopup="true" aria-expanded="false" role="button" tabindex="0" id="country-switcher-menu">
                            <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej21 _18ybopc4 i7p5ej7">
                              <div class="gepq850 eihuid5b eihuidh7 eihuidgr eihuidgj eihuidn _16qi62m2b">
                                <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87" aria-hidden="true">
                                      <path d="M12 1C7.6 1 4 4.6 4 9c0 4.1 6.5 12.6 7.2 13.6.2.2.5.4.8.4s.6-.1.8-.4c.7-1 7.2-9.5 7.2-13.6 0-4.4-3.6-8-8-8zm0 19.3c-2.2-3-6-8.8-6-11.3 0-3.3 2.7-6 6-6s6 2.7 6 6c0 2.5-3.8 8.3-6 11.3z"></path>
                                      <path d="M12 5c-1.7 0-3 1.3-3 3s1.3 3 3 3 3-1.3 3-3-1.3-3-3-3zm0 4c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1z"></path>
                                    </svg>
                                  </span>
                                </div>
                                <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidaz">Philippines</div>
                                <div class="gepq850 eihuid4z eihuidr eihuidp eihuidi3 eihuidaz">
                                  <span class="gepq850 eihuid57">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" xml:space="preserve" focusable="false" fill="currentColor" width="16" height="16" class="gepq850 _1wnzuop0 eihuid57 eihuid5f _74wkf80 _74wkf82 _74wkf83 _74wkf87 _1wnzuop2" aria-hidden="true">
                                      <path d="M20.7 7.3c-.4-.4-1-.4-1.4 0L12 14.6 4.7 7.3c-.4-.4-1-.4-1.4 0s-.4 1 0 1.4l8 8c.2.2.5.3.7.3s.5-.1.7-.3l8-8c.4-.4.4-1 0-1.4z"></path>
                                    </svg>
                                  </span>
                                </div>
                              </div>
                            </span>
                          </button>
                        </div>
                      </div>
                    </div>
                    <div class="gepq850 eihuid5b eihuidhf eihuidha eihuidgi eihuid6z eihuidhn _5qpagp3">
                      <div class="gepq850 eihuid5b eihuidhf eihuidha eihuidgr eihuidn _16qi62m2j _16qi62m39">
                        <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidbv eihuidbi eihuid83 eihuid8m _17256u60">
                          <a href="/terms" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="terms &amp; conditions" target="_self">
                            <div class="gepq850 eihuid5b eihuidgj">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej4">
                                <span class="gepq850 _12f6icc1">Terms &amp; conditions</span>
                              </span>
                            </div>
                          </a>
                        </div>
                        <div class="gepq850 eihuid4z eihuidr eihuidhv eihuidhz eihuidbv eihuidbi eihuid83 eihuid8m _17256u60">
                          <a href="/security-privacy" class="gepq850 gepq85f  gepq850 gepq85f _12f6icc0" data-automation="security &amp; privacy" target="_self">
                            <div class="gepq850 eihuid5b eihuidgj">
                              <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej4">
                                <span class="gepq850 _12f6icc1">Security &amp; Privacy</span>
                              </span>
                            </div>
                          </a>
                        </div>
                      </div>
                      <span class="gepq850 eihuid4z i7p5ej0 i7p5ej1 i7p5ej22 _18ybopc4 i7p5ej4">Copyright © 2025, Jobstreet</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </div>
      <script>
        window.__staticRouterHydrationData = JSON.parse("{\\"
          loaderData\\ ":{\\"
          0\\ ":null,\\" / job / : jobId - en\\ ":null},\\"
          actionData\\ ":null,\\"
          errors\\ ":null}");
      </script>
    </div>\n <script data-automation="server-state">
      \
      n window.SEEK_CONFIG = {
        "SEEK_API_V5_COUNT_ENDPOINT_PUBLIC": "\\u002Fapi\\u002Fjobsearch\\u002Fv5\\u002Fcount",
        "SEEK_API_V5_COUNTS_ENDPOINT_PUBLIC": "https:\\u002F\\u002Fjobsearch-api-ts.cloud.seek.com.au\\u002Fv5\\u002Fcounts",
        "SEEK_METRICS_ENABLED": "true",
        "SEEK_STATSD_HOST": "127.0.0.1",
        "SEEK_API_V5_COUNT_ENDPOINT_AUTHED": "http:\\u002F\\u002Fjobsearch-api-ts.int-cloud.seek.com.au\\u002Fv5\\u002Fme\\u002Fcount",
        "SEEK_API_V5_SEARCH_ENDPOINT_PUBLIC_AUTHED": "\\u002Fapi\\u002Fjobsearch\\u002Fv5\\u002Fme\\u002Fsearch",
        "SEEK_GRAPHQL_ENDPOINT_PUBLIC": "\\u002Fgraphql",
        "SEEK_API_V5_COUNTS_ENDPOINT_AUTHED": "https:\\u002F\\u002Fjobsearch-api-ts.cloud.seek.com.au\\u002Fv5\\u002Fme\\u002Fcounts",
        "SEEK_API_V5_SEARCH_ENDPOINT": "http:\\u002F\\u002Fjobsearch-api-ts.int-cloud.seek.com.au\\u002Fv5\\u002Fsearch",
        "SEEK_DATADOG_BROWSER_LOGS_ENABLED": "true",
        "SEEK_JOB_TRACKING_ENABLED": "true",
        "SEEK_DATADOG_RUM_ENABLED": "true",
        "SEEK_STATSD_PORT": "8125",
        "SEEK_ADVERTS_ENABLED": "true",
        "SEEK_METRICS_PREFIX": "discover",
        "SEEK_SAVED_SEARCHES_PATH": "\\u002Fmy-activity\\u002Fsaved-searches",
        "SEEK_GRAPHQL_ENDPOINT": "http:\\u002F\\u002Fcandidate-graphql.int-cloud.seek.com.au\\u002Fgraphql",
        "SEEK_LOG_LEVEL": "info",
        "SEEK_ANALYTICS_ENABLED": "true",
        "SEEK_API_V5_COUNT_ENDPOINT_PUBLIC_AUTHED": "\\u002Fapi\\u002Fjobsearch\\u002Fv5\\u002Fme\\u002Fcount",
        "SEEK_SIGN_IN_PATH": "\\u002Foauth\\u002Flogin\\u002F",
        "SEEK_VERSION": "gantry-fa6a54d.45258",
        "SEEK_ENVIRONMENT": "production",
        "SEEK_METRICS_HOST": "https:\\u002F\\u002Fdiscover-metrics.cloud.seek.com.au",
        "SEEK_SIGN_IN_REG_URL_PREFIX": "",
        "SEEK_JOB_TRACKING_URL": "https:\\u002F\\u002Ftracking.engineering.cloud.seek.com.au\\u002Fa.png",
        "SEEK_SSA_ENABLED": "true",
        "SEEK_API_V5_SEARCH_ENDPOINT_PUBLIC": "\\u002Fapi\\u002Fjobsearch\\u002Fv5\\u002Fsearch",
        "SEEK_CAREER_FEED_GEN_AI_API_URL": "https:\\u002F\\u002Fdiscover-career-feed-genai-api.cloud.seek.com.au",
        "SEEK_GOOGLE_ONE_TAP_ENABLED": "true",
        "SEEK_HOTJAR_ENABLED": "true",
        "SEEK_SOL_TRACKING_ENDPOINT": "https:\\u002F\\u002Fweb.aips-sol.com",
        "SEEK_DATADOG_BROWSER_LOG_METADATA": "{\\"
        clientToken\\ ":\\"
        pub0589d1a5a97fa3b4c01369efda094472\\ "}",
        "SEEK_DATADOG_RUM_APPLICATION_METADATA": "{\\"
        applicationId\\ ":\\"
        cfba1d30 - 54e3 - 4 a25 - b2ab - 4 af0c589515a\\ ",\\"
        clientToken\\ ":\\"
        pub5a88787c972861476b14b524d64495c7\\ "}",
        "NODE_ENV": "production",
        "SEEK_REGISTER_PATH": "\\u002Foauth\\u002Fregister\\u002F",
        "SEEK_API_V5_COUNT_ENDPOINT": "http:\\u002F\\u002Fjobsearch-api-ts.int-cloud.seek.com.au\\u002Fv5\\u002Fcount"
      };\
      n window.SEEK_REDUX_DATA = {
        "appConfig": {
          "brand": "jobstreet",
          "site": "candidate-jobstreet-ph",
          "language": "en",
          "country": "PH",
          "zone": "asia-6",
          "zoneFeatures": {
            "NUDGE_ENABLED": false,
            "HOMEPAGE_BANNER_TYPE": "GENERIC_ASIA_BANNER",
            "SEARCH_PAGE_SIZE": 32,
            "SHOW_FLOATING_SAVE_SEARCH": false,
            "AUTO_SELECT_SPLIT_VIEW_FIRST_JOB": false,
            "SHOW_MARKETING_AD_SPLIT_VIEW": true,
            "LMIS_ENABLED": true,
            "BEHAVIOURAL_CUES_ENABLED": true,
            "BEHAVIOURAL_CUES_FILTERS_ENABLED": false,
            "LOGGED_OUT_RECS": false,
            "REMOTE_SEARCH_FILTER": true,
            "REMOTE_SEARCH_FILTER_NEW_LABEL": true,
            "DYNAMIC_PILLS": false,
            "KEYWORD_AUTOSUGGEST_V2": true,
            "NEW_JOB_CARD_DENSITY": false,
            "ENABLE_VERIFIED_HIRER_BADGE": true,
            "SERP_JOBCARD_INFO_DENSITY_1": false,
            "MATCHED_QUALITIES": false,
            "ENTRY_LEVEL_BADGE": true
          },
          "locale": "en-PH"
        },
        "experiments": {
          "matchedQualities": {
            "name": "matched_qualities",
            "num": 72,
            "variation": {
              "name": "control",
              "index": "0"
            }
          }
        },
        "featureFlags": {
          "behaviouralCues": true,
          "behaviouralCuesFilters": false,
          "branchBannerPreview": false,
          "isBranchEnabledFlag": true,
          "showHomepageBanner": false,
          "showFloatingSaveSearch": false,
          "autoSelectSplitViewFirstJob": false,
          "showMarketingAdSplitView": true,
          "loggedOutRecs": false,
          "remoteSearchFilter": true,
          "remoteSearchFilterNewLabel": true,
          "dynamicPills": false,
          "dynamicPillsV2": false,
          "stickySearchBar": false,
          "refineBarV2": false,
          "keywordAutosuggestV2": true,
          "hirerVerifiedBadge": true,
          "serpJobCardInfoDensity1": false,
          "newJobCardDensity": false,
          "hideCompanyLogo": false,
          "matchedQualities": false,
          "hideApplyButtonOnPrivateAdvertiser": true,
          "entryLevelBadge": true,
          "homepageGoogleAds": false,
          "homepageRecsBadgingStrongApplicant": false,
          "homepageRecsBadgingExpiringSoonEarlyApplicant": false,
          "homepageJobCardDensity": false,
          "homepageGenAIChat": false,
          "homepageGenAIChatAltModel": false,
          "showJobDisplayTags": true,
          "competitivePlacement": false,
          "savedRecentSearches": false
        },
        "jobdetails": {
          "fraudReport": {},
          "jobPending": false,
          "result": {
            "__typename": "JobDetails",
            "learningInsights": {
              "__typename": "LearningInsights",
              "analytics": {
                "title": "Mid-Level Frontend Developer",
                "landingPage": "CA Role Salary",
                "resultType": "CareerAdviceSalaryTeaserSearch:frontend-developer",
                "entity": "career",
                "encoded": "title:Mid-Level Frontend Developer;landingPage:CA Role Salary;resultType:CareerAdviceSalaryTeaserSearch:frontend-developer;entity:career"
              },
              "content": "\\u003Cstyle\\u003E\\n  \\u002F* capsize font, don\'t change *\\u002F\\n  .capsize-heading4 {\\n    font-size: 20px;\\n    line-height: 24.66px;\\n    display: block;\\n    font-weight: 500;\\n  }\\n\\n  .capsize-heading4::before {\\n    content: \'\';\\n    margin-bottom: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-heading4::after {\\n    content: \'\';\\n    margin-top: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText {\\n    font-size: 16px;\\n    line-height: 24.528px;\\n    display: block;\\n  }\\n\\n  .capsize-standardText::before {\\n    content: \'\';\\n    margin-bottom: -0.375em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText::after {\\n    content: \'\';\\n    margin-top: -0.375em;\\n    display: table;\\n  }\\n\\n  @media only screen and (min-width: 740px) {\\n    .capsize-heading3 {\\n      font-size: 24px;\\n      line-height: 29.792px;\\n    }\\n\\n    .capsize-heading3::before {\\n      content: \'\';\\n      margin-bottom: -0.2292em;\\n      display: table;\\n    }\\n\\n    .capsize-heading3::after {\\n      content: \'\';\\n      margin-top: -0.2292em;\\n      display: table;\\n    }\\n  }\\n  \\u002F* end of capsize *\\u002F\\n\\n  \\u002F* LMIS css start here*\\u002F\\n  .lmis-root {\\n    margin: -32px;\\n    padding: 32px;\\n    font-family: SeekSans, \'SeekSans Fallback\', Arial, sans-serif;\\n    background: #beeff3;\\n    border-radius: 16px;\\n    color: #2e3849;\\n  }\\n\\n  .lmis-title {\\n    margin-bottom: 8px;\\n  }\\n\\n  .lmis-cta {\\n    min-height: 48px;\\n    display: flex;\\n    align-items: center;\\n    color: #2e3849;\\n    text-decoration: none;\\n  }\\n\\n  .lmis-cta-text {\\n    margin-right: 4px;\\n    font-weight: 500;\\n  }\\n\\n  .lmis-teaser-image {\\n    max-width: 96px;\\n  }\\n\\n  @media only screen and (min-width: 992px) {\\n    .lmis-root {\\n      margin: -48px;\\n    }\\n\\n    .lmis-wrapper {\\n      display: flex;\\n      flex-direction: row-reverse;\\n      justify-content: space-between;\\n      align-items: center;\\n    }\\n  }\\n\\u003C\\u002Fstyle\\u003E\\n\\n\\u003Cdiv class=\\"
              lmis - root\\ "\\u003E\\n  \\u003Cdiv class=\\"
              lmis - wrapper\\ "\\u003E\\n    \\u003Cdiv class=\\"
              lmis - teaser - image\\ "\\u003E\\n      \\u003Cimg\\n        src=\\"
              https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Fgirl_comparing_salaries.svg\\ "\\n        alt=\\"
              salary teaser image\\ "\\n      \\u002F\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n    \\u003Cdiv class=\\"
              lmis - content\\ "\\u003E\\n      \\u003Cdiv class=\\"
              capsize - heading4 lmis - title\\ "\\u003EWhat can I earn as a Front End Developer\\u003C\\u002Fdiv\\u003E\\n      \\u003Ca\\n        class=\\"
              lmis - cta\\ "\\n        href=\\"
              https: \\u002F\ \u002Fph.jobstreet.com\ \u002Fcareer - advice\ \u002Frole\ \u002Ffrontend - developer\ \u002Fsalary ? campaigncode = lrn : skj: sklm: cg: jbd: alpha\\ "\\n        target=\\"
              _blank\\ "\\n      \\u003E\\n        \\u003Cspan class=\\"
              capsize - standardText lmis - cta - text\\ "\\u003ESee more detailed salary information\\u003C\\u002Fspan\\u003E\\n        \\u003Cimg\\n          src=\\"
              https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Farrow_right.svg\\ "\\n          alt=\\"
              salary teaser link arrow\\ "\\n        \\u002F\\u003E\\n      \\u003C\\u002Fa\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n  \\u003C\\u002Fdiv\\u003E\\n\\u003C\\u002Fdiv\\u003E\\n"
            },
            "gfjInfo": {
              "__typename": "GFJInfo",
              "location": {
                "__typename": "GFJLocation",
                "countryCode": "PH",
                "country": "Philippines",
                "suburb": null,
                "region": null,
                "state": "Metro Manila",
                "postcode": null
              },
              "workTypes": {
                "__typename": "GFJWorkTypes",
                "label": ["FULL_TIME"]
              }
            },
            "seoInfo": {
              "__typename": "SEOInfo",
              "normalisedRoleTitle": "Front End Developer",
              "workType": "242",
              "classification": ["6281"],
              "subClassification": ["6287"],
              "where": "Manila City Metro Manila",
              "broaderLocationName": "Manila City",
              "normalisedOrganisationName": "CoDev"
            },
            "job": {
              "__typename": "Job",
              "sourceZone": "asia-6",
              "tracking": {
                "__typename": "JobTracking",
                "adProductType": null,
                "classificationInfo": {
                  "__typename": "JobTrackingClassificationInfo",
                  "classificationId": "6281",
                  "classification": "Information & Communication Technology",
                  "subClassificationId": "6287",
                  "subClassification": "Developers\\u002FProgrammers"
                },
                "hasRoleRequirements": true,
                "isPrivateAdvertiser": false,
                "locationInfo": {
                  "__typename": "JobTrackingLocationInfo",
                  "area": null,
                  "location": "Manila City",
                  "locationIds": ["2061106"]
                },
                "workTypeIds": "242",
                "postedTime": "3d ago"
              },
              "id": "82411354",
              "title": "Mid-Level Frontend Developer",
              "phoneNumber": null,
              "isExpired": false,
              "expiresAt": {
                "__typename": "SeekDateTime",
                "dateTimeUtc": "2025-03-29T12:59:59.999Z"
              },
              "isLinkOut": false,
              "contactMatches": [],
              "isVerified": true,
              "abstract": "Will work on a web-based application that allows users to input metropolitan areas and healthcare billing codes",
              "content": "\\u003Cp\\u003EThe Front-End Developer will work on the \\u003Cstrong\\u003Ehealthcare price benchmarking tool\\u003C\\u002Fstrong\\u003E, a web-based application that allows users to input metropolitan areas and healthcare billing codes (CPT codes) to receive relevant benchmark pricing data. This tool will include:\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EA \\u003Cstrong\\u003Esearchable interface\\u003C\\u002Fstrong\\u003E for users to input and retrieve pricing data\\u003C\\u002Fli\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003EData visualization elements\\u003C\\u002Fstrong\\u003E such as percentile ranges, regional heat maps, and interactive graphs\\u003C\\u002Fli\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003EExport functionality\\u003C\\u002Fstrong\\u003E for reports and datasets\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003EThe back-end, including data scrubbing and storage, will be handled by Simple Healthcare, with the front-end focused on presenting and visualizing this information.\\u003C\\u002Fp\\u003E\\u003Cp\\u003EThis is the first step in a broader roadmap, with a more \\u003Cstrong\\u003Ecomplex BI (Business Intelligence) dashboard\\u003C\\u002Fstrong\\u003E planned for future development in 2026.\\u003C\\u002Fp\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EResponsibilities\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EDevelop a \\u003Cstrong\\u003Escalable, user-friendly front-end\\u003C\\u002Fstrong\\u003E for the price benchmarking tool\\u003C\\u002Fli\\u003E\\u003Cli\\u003EImplement \\u003Cstrong\\u003Esearch and filtering functionalities\\u003C\\u002Fstrong\\u003E for healthcare pricing data\\u003C\\u002Fli\\u003E\\u003Cli\\u003EDevelop \\u003Cstrong\\u003Edata visualization elements\\u003C\\u002Fstrong\\u003E (heat maps, percentile charts, interactive tables)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EOptimize front-end performance for \\u003Cstrong\\u003Espeed and accessibility\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003ECollaborate with a \\u003Cstrong\\u003EUI\\u002FUX designer\\u003C\\u002Fstrong\\u003E to implement user-friendly designs\\u003C\\u002Fli\\u003E\\u003Cli\\u003EEnsure the front-end integrates seamlessly with the back-end APIs\\u003C\\u002Fli\\u003E\\u003Cli\\u003EMaintain \\u003Cstrong\\u003Eclean, scalable, and well-documented\\u003C\\u002Fstrong\\u003E code for future product expansion\\u003C\\u002Fli\\u003E\\u003Cli\\u003EWork closely with stakeholders to gather and refine requirements\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EQualifications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EMust-Have:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003E4+ years\\u003C\\u002Fstrong\\u003E of front-end development experience\\u003C\\u002Fli\\u003E\\u003Cli\\u003EStrong proficiency in \\u003Cstrong\\u003EReact.js\\u003C\\u002Fstrong\\u003E (preferred) or \\u003Cstrong\\u003EVue.js\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003ENext.js\\u003C\\u002Fstrong\\u003E (or similar SSR frameworks)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EStrong knowledge of \\u003Cstrong\\u003EJavaScript, TypeScript, HTML5, CSS3\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003ERESTful APIs and JSON data handling\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EFamiliarity with \\u003Cstrong\\u003Edata visualization libraries\\u003C\\u002Fstrong\\u003E (D3.js, Chart.js, Highcharts, or similar)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EKnowledge of \\u003Cstrong\\u003Estate management\\u003C\\u002Fstrong\\u003E (Redux, Context API)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EUnderstanding of \\u003Cstrong\\u003Eresponsive and accessible web design\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003ENice-to-Have:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003EBI tools\\u003C\\u002Fstrong\\u003E or \\u003Cstrong\\u003Edata-heavy applications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EFamiliarity with \\u003Cstrong\\u003EMaterial UI, Tailwind CSS, or other UI libraries\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003Ehealthcare or financial tech applications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EBasic understanding of \\u003Cstrong\\u003Ebackend technologies (Node.js, Express.js, PostgreSQL\\u002FMongoDB)\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExposure to \\u003Cstrong\\u003EReact Native\\u003C\\u002Fstrong\\u003E for potential future mobile development\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EAdditional Skill\\u002F Ideal:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EBasic knowledge of \\u003Cstrong\\u003ESQL databases\\u003C\\u002Fstrong\\u003E.\\u003C\\u002Fli\\u003E\\u003Cli\\u003EHas understanding of \\u003Cstrong\\u003EUI\\u002FUX principles\\u003C\\u002Fstrong\\u003E and front-end performance optimization.\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E",
              "status": "Active",
              "listedAt": {
                "__typename": "SeekDateTime",
                "label": "3d ago",
                "dateTimeUtc": "2025-02-27T08:15:14.991Z"
              },
              "salary": {
                "__typename": "JobSalary",
                "currencyLabel": null,
                "label": "₱45,000 – ₱55,000 per month"
              },
              "shareLink": "https:\\u002F\\u002Fph.jobstreet.com\\u002Fjob\\u002F82411354?tracking=SHR-WEB-SharedJob-asia-6",
              "workTypes": {
                "__typename": "JobWorkTypes",
                "label": "Full time"
              },
              "advertiser": {
                "__typename": "Advertiser",
                "id": "60276460",
                "name": "Complete Development (CoDev)",
                "isVerified": true,
                "registrationDate": {
                  "__typename": "SeekDateTime",
                  "dateTimeUtc": "2010-07-30T08:08:58.843Z"
                }
              },
              "location": {
                "__typename": "LocationInfo",
                "label": "Manila City, Metro Manila"
              },
              "classifications": [{
                "__typename": "ClassificationInfo",
                "label": "Developers\\u002FProgrammers (Information & Communication Technology)"
              }],
              "products": {
                "__typename": "JobProducts",
                "branding": {
                  "__typename": "JobProductBranding",
                  "id": "9d1d0514-d034-4e58-87c9-bde0b8073d60.1",
                  "cover": {
                    "__typename": "JobProductBrandingImage",
                    "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffee0917cda33adaff32bfc9def19be674d8dc397\\u002F77dcaec03772f01fbffa028a4883e2282ec6925b"
                  },
                  "thumbnailCover": {
                    "__typename": "JobProductBrandingImage",
                    "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffee0917cda33adaff32bfc9def19be674d8dc397\\u002F786c2c2276de5a15f87d4892dcad8b97e324f68d"
                  },
                  "logo": {
                    "__typename": "JobProductBrandingImage",
                    "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Fc9c08ac77f444302c37bfa8a82c13639a1a960e3\\u002Fee4dce1061f3f616224767ad58cb2fc751b8d2dc"
                  }
                },
                "bullets": ["Urgent hiring", "Work from home", "Great benefits with free online courses, HMO benefit and more"],
                "questionnaire": {
                  "__typename": "JobQuestionnaire",
                  "questions": ["What\'s your expected monthly basic salary?", "Which of the following types of qualifications do you have?", "How many years\' experience do you have as a frontend software developer?", "Which of the following programming languages are you experienced in?", "How many years of front end development experience do you have?", "Which of the following front end development libraries and frameworks are you proficient in?", "How would you rate your English language skills?"]
                },
                "video": {
                  "__typename": "VideoProduct",
                  "url": "https:\\u002F\\u002Fwww.youtube.com\\u002Fembed\\u002FNFJmx-M_M4U?t=70s",
                  "position": "BOTTOM"
                },
                "displayTags": []
              }
            },
            "companyProfile": {
              "__typename": "CompanyProfile",
              "id": "168553861426412",
              "name": "CoDev",
              "companyNameSlug": "codev-168553861426412",
              "shouldDisplayReviews": true,
              "branding": {
                "__typename": "Branding",
                "logo": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffce76906ed15a97d1c1f92c5b6b3125153f11c0e\\u002Fee4dce1061f3f616224767ad58cb2fc751b8d2dc"
              },
              "overview": {
                "__typename": "Overview",
                "description": {
                  "__typename": "Description",
                  "paragraphs": ["Founded 14 years ago, CoDev connects highly-skilled developers from emerging nations such as the Philippines with small and medium-sized businesses in the US and Canada.", "Our goal is to empower coders from the world’s most exciting markets, so they can partner with US and Canadian visionaries to create groundbreaking products for the online world.", "Why Join CoDev:", "Competitive Salary & Benefits", "Fun Culture & Working Environment", "Upskilling and Personal Growth", "Work with U.S. Counterparts", "Our Core Values:", "Trustworthy", "Reliable", "Driven", "Positive", "Kind"]
                },
                "industry": "Information & Communication Technology",
                "size": {
                  "__typename": "CompanySize",
                  "description": "101-1,000 employees"
                },
                "website": {
                  "__typename": "Website",
                  "url": "https:\\u002F\\u002Fwww.codev.com\\u002Fcareers\\u002F"
                }
              },
              "reviewsSummary": {
                "__typename": "ReviewsSummary",
                "overallRating": {
                  "__typename": "OverallRating",
                  "numberOfReviews": {
                    "__typename": "NumberOfReviews",
                    "value": 16
                  },
                  "value": 4.1875
                }
              },
              "perksAndBenefits": [{
                "__typename": "PerkAndBenefit",
                "title": "Life Insurance, Health\\u002FHMO Insurance +2 Dependents"
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Miscellaneous allowance"
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Annual Salary Increase "
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Remote Work Flexibility"
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Company Employee Morale Activities"
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Upskill \\u002F Online Training"
              }, {
                "__typename": "PerkAndBenefit",
                "title": "Company Issued Machine"
              }]
            },
            "companySearchUrl": "https:\\u002F\\u002Fph.jobstreet.com\\u002FCoDev-jobs\\u002Fat-this-company",
            "companyTags": [],
            "restrictedApplication": {
              "__typename": "JobDetailsRestrictedApplication",
              "label": null
            },
            "sourcr": null
          },
          "pageLoadedCount": 0,
          "personalised": null,
          "xRealIp": "180.190.115.57",
          "jobDetailsViewedCorrelationId": "e73f839e-7e2d-4127-a375-9b8b7fa921c2",
          "error": false
        },
        "recentSearches": {
          "searches": []
        },
        "lmis": {
          "SRP": {
            "content": ""
          },
          "JDP": {
            "content": "\\u003Cstyle\\u003E\\n  \\u002F* capsize font, don\'t change *\\u002F\\n  .capsize-heading4 {\\n    font-size: 20px;\\n    line-height: 24.66px;\\n    display: block;\\n    font-weight: 500;\\n  }\\n\\n  .capsize-heading4::before {\\n    content: \'\';\\n    margin-bottom: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-heading4::after {\\n    content: \'\';\\n    margin-top: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText {\\n    font-size: 16px;\\n    line-height: 24.528px;\\n    display: block;\\n  }\\n\\n  .capsize-standardText::before {\\n    content: \'\';\\n    margin-bottom: -0.375em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText::after {\\n    content: \'\';\\n    margin-top: -0.375em;\\n    display: table;\\n  }\\n\\n  @media only screen and (min-width: 740px) {\\n    .capsize-heading3 {\\n      font-size: 24px;\\n      line-height: 29.792px;\\n    }\\n\\n    .capsize-heading3::before {\\n      content: \'\';\\n      margin-bottom: -0.2292em;\\n      display: table;\\n    }\\n\\n    .capsize-heading3::after {\\n      content: \'\';\\n      margin-top: -0.2292em;\\n      display: table;\\n    }\\n  }\\n  \\u002F* end of capsize *\\u002F\\n\\n  \\u002F* LMIS css start here*\\u002F\\n  .lmis-root {\\n    margin: -32px;\\n    padding: 32px;\\n    font-family: SeekSans, \'SeekSans Fallback\', Arial, sans-serif;\\n    background: #beeff3;\\n    border-radius: 16px;\\n    color: #2e3849;\\n  }\\n\\n  .lmis-title {\\n    margin-bottom: 8px;\\n  }\\n\\n  .lmis-cta {\\n    min-height: 48px;\\n    display: flex;\\n    align-items: center;\\n    color: #2e3849;\\n    text-decoration: none;\\n  }\\n\\n  .lmis-cta-text {\\n    margin-right: 4px;\\n    font-weight: 500;\\n  }\\n\\n  .lmis-teaser-image {\\n    max-width: 96px;\\n  }\\n\\n  @media only screen and (min-width: 992px) {\\n    .lmis-root {\\n      margin: -48px;\\n    }\\n\\n    .lmis-wrapper {\\n      display: flex;\\n      flex-direction: row-reverse;\\n      justify-content: space-between;\\n      align-items: center;\\n    }\\n  }\\n\\u003C\\u002Fstyle\\u003E\\n\\n\\u003Cdiv class=\\"
            lmis - root\\ "\\u003E\\n  \\u003Cdiv class=\\"
            lmis - wrapper\\ "\\u003E\\n    \\u003Cdiv class=\\"
            lmis - teaser - image\\ "\\u003E\\n      \\u003Cimg\\n        src=\\"
            https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Fgirl_comparing_salaries.svg\\ "\\n        alt=\\"
            salary teaser image\\ "\\n      \\u002F\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n    \\u003Cdiv class=\\"
            lmis - content\\ "\\u003E\\n      \\u003Cdiv class=\\"
            capsize - heading4 lmis - title\\ "\\u003EWhat can I earn as a Front End Developer\\u003C\\u002Fdiv\\u003E\\n      \\u003Ca\\n        class=\\"
            lmis - cta\\ "\\n        href=\\"
            https: \\u002F\ \u002Fph.jobstreet.com\ \u002Fcareer - advice\ \u002Frole\ \u002Ffrontend - developer\ \u002Fsalary ? campaigncode = lrn : skj: sklm: cg: jbd: alpha\\ "\\n        target=\\"
            _blank\\ "\\n      \\u003E\\n        \\u003Cspan class=\\"
            capsize - standardText lmis - cta - text\\ "\\u003ESee more detailed salary information\\u003C\\u002Fspan\\u003E\\n        \\u003Cimg\\n          src=\\"
            https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Farrow_right.svg\\ "\\n          alt=\\"
            salary teaser link arrow\\ "\\n        \\u002F\\u003E\\n      \\u003C\\u002Fa\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n  \\u003C\\u002Fdiv\\u003E\\n\\u003C\\u002Fdiv\\u003E\\n",
            "lmisSnippet": {
              "title": "Mid-Level Frontend Developer",
              "landingPage": "CA Role Salary",
              "resultType": "CareerAdviceSalaryTeaserSearch:frontend-developer",
              "entity": "career",
              "encoded": "title:Mid-Level Frontend Developer;landingPage:CA Role Salary;resultType:CareerAdviceSalaryTeaserSearch:frontend-developer;entity:career"
            },
            "key": "JDP"
          }
        },
        "location": {
          "url": "http:\\u002F\\u002Fph.jobstreet.com\\u002Fjob\\u002F82411354",
          "prevPathname": "",
          "query": {},
          "pageNumber": 1,
          "isHomepage": false,
          "pathname": "\\u002Fjob\\u002F82411354",
          "hostname": "ph.jobstreet.com",
          "href": "http:\\u002F\\u002Fph.jobstreet.com\\u002Fjob\\u002F82411354",
          "port": "",
          "protocol": "http:",
          "hash": ""
        },
        "results": {
          "results": null,
          "isLoading": false,
          "isError": false,
          "source": "",
          "locationWhere": null,
          "title": "",
          "totalCountNewToYou": null,
          "totalCount": null,
          "totalPages": 0,
          "jobIds": [],
          "lastPage": 0,
          "sortMode": null,
          "solReferenceKeys": [],
          "hidden": false,
          "relatedSearches": [],
          "facets": {},
          "titleTagShortLocationName": "",
          "titleTagTestTitle": "",
          "searchSource": ""
        },
        "saveSearch": {
          "emailAddress": "",
          "errorMessage": "",
          "saveable": true,
          "status": "UNSAVED"
        },
        "search": {
          "baseKeywords": "",
          "keywordsField": "",
          "whereField": "",
          "filtersExpanded": false,
          "query": {},
          "lastQuery": {},
          "hasLoadedCounts": false,
          "refinements": {},
          "poeaCountryPicker": false,
          "poeaCountryPickerFromNavigation": false,
          "dynamicSearchBarHeight": 177,
          "dynamicSearchBarType": "expanded",
          "dynamicSearchBarConfiguration": "CLASSIC"
        },
        "seo": {
          "canonicalUrl": "\\u002Fjob\\u002F82411354",
          "partners": {
            "canCrossLink": false
          }
        },
        "user": {
          "authenticated": false,
          "userClientId": "c81773e2-75a4-40e5-ae80-c4f7986d1551",
          "sessionId": "c81773e2-75a4-40e5-ae80-c4f7986d1551",
          "testHeaders": {},
          "solId": "9b27e9e1-16d2-4f7a-9abd-b8880f8c1acc",
          "serverAuthenticated": false
        },
        "banner": {
          "template": {
            "items": []
          },
          "error": null
        },
        "@@redux-hotjar-state": []
      };\
      n window.SEEK_APP_CONFIG = {
        "zone": "asia-6",
        "defaultLocale": "en-PH",
        "availableLocales": ["en-PH"],
        "timedBanners": {},
        "zoneFeatures": {
          "NUDGE_ENABLED": false,
          "HOMEPAGE_BANNER_TYPE": "GENERIC_ASIA_BANNER",
          "SEARCH_PAGE_SIZE": 32,
          "SHOW_FLOATING_SAVE_SEARCH": false,
          "AUTO_SELECT_SPLIT_VIEW_FIRST_JOB": false,
          "SHOW_MARKETING_AD_SPLIT_VIEW": true,
          "LMIS_ENABLED": true,
          "BEHAVIOURAL_CUES_ENABLED": true,
          "BEHAVIOURAL_CUES_FILTERS_ENABLED": false,
          "LOGGED_OUT_RECS": false,
          "REMOTE_SEARCH_FILTER": true,
          "REMOTE_SEARCH_FILTER_NEW_LABEL": true,
          "DYNAMIC_PILLS": false,
          "KEYWORD_AUTOSUGGEST_V2": true,
          "NEW_JOB_CARD_DENSITY": false,
          "ENABLE_VERIFIED_HIRER_BADGE": true,
          "SERP_JOBCARD_INFO_DENSITY_1": false,
          "MATCHED_QUALITIES": false,
          "ENTRY_LEVEL_BADGE": true
        },
        "BRANCH_IO_KEY": "key_live_keVW4JKlS0jBJLPfSRNUJnhhtElmK541",
        "GPT_ACCOUNT_ID": "23204624037",
        "brand": "jobstreet",
        "locale": "en-PH",
        "site": "candidate-jobstreet-ph",
        "country": "PH",
        "language": "en"
      };\
      n window.SK_DL = {
        "brand": "jobstreet",
        "isLoggedIn": false,
        "loginId": "NULL",
        "siteCountry": "ph",
        "siteLanguage": "en",
        "siteSection": "discover",
        "zone": "asia-6",
        "experiments": [{
          "id": "matched_qualities",
          "variant": "0",
          "num": 72
        }],
        "hubbleExperiments": {
          "matched_qualities": {
            "name": "matched_qualities",
            "participantStatus": "enrolled",
            "variant": "0"
          }
        },
        "seekAdvertiserId": "60276460",
        "advertiserName": "Complete Development (CoDev)",
        "companyId": "168553861426412",
        "companyName": "CoDev",
        "companyRating": "4.1875",
        "isPrivateAdvertiser": false,
        "jobId": "82411354",
        "jobTitle": "Mid-Level Frontend Developer",
        "jobIsLinkOut": false,
        "jobHasRoleRequirements": true,
        "jobStatus": "active",
        "jobSalary": "₱45,000 – ₱55,000 per month",
        "jobPostedTime": "3d ago",
        "jobLocation": "Manila City",
        "jobWorkTypeIds": "242",
        "jobWhereId": "2061106",
        "jobClassification": "Information & Communication Technology",
        "jobClassificationId": "6281",
        "jobSubClassification": "Developers\\u002FProgrammers",
        "jobSubClassificationId": "6287"
      };\
      n window.SEEK_APOLLO_DATA = {
        "Advertiser:60276460": {
          "__typename": "Advertiser",
          "id": "60276460",
          "name({\\"
          locale\\ ":\\"
          en - PH\\ "})": "Complete Development (CoDev)",
          "isVerified": true,
          "registrationDate": {
            "__typename": "SeekDateTime",
            "dateTimeUtc": "2010-07-30T08:08:58.843Z"
          }
        },
        "JobProductBranding:9d1d0514-d034-4e58-87c9-bde0b8073d60.1": {
          "__typename": "JobProductBranding",
          "id": "9d1d0514-d034-4e58-87c9-bde0b8073d60.1",
          "cover": {
            "__typename": "JobProductBrandingImage",
            "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffee0917cda33adaff32bfc9def19be674d8dc397\\u002F77dcaec03772f01fbffa028a4883e2282ec6925b"
          },
          "cover({\\"
          isThumbnail\\ ":true})": {
            "__typename": "JobProductBrandingImage",
            "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffee0917cda33adaff32bfc9def19be674d8dc397\\u002F786c2c2276de5a15f87d4892dcad8b97e324f68d"
          },
          "logo": {
            "__typename": "JobProductBrandingImage",
            "url": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Fc9c08ac77f444302c37bfa8a82c13639a1a960e3\\u002Fee4dce1061f3f616224767ad58cb2fc751b8d2dc"
          }
        },
        "CompanyProfile:168553861426412": {
          "__typename": "CompanyProfile",
          "id": "168553861426412",
          "name": "CoDev",
          "companyNameSlug": "codev-168553861426412",
          "shouldDisplayReviews": true,
          "branding": {
            "__typename": "Branding",
            "logo": "https:\\u002F\\u002Fimage-service-cdn.seek.com.au\\u002Ffce76906ed15a97d1c1f92c5b6b3125153f11c0e\\u002Fee4dce1061f3f616224767ad58cb2fc751b8d2dc"
          },
          "overview": {
            "__typename": "Overview",
            "description": {
              "__typename": "Description",
              "paragraphs": ["Founded 14 years ago, CoDev connects highly-skilled developers from emerging nations such as the Philippines with small and medium-sized businesses in the US and Canada.", "Our goal is to empower coders from the world’s most exciting markets, so they can partner with US and Canadian visionaries to create groundbreaking products for the online world.", "Why Join CoDev:", "Competitive Salary & Benefits", "Fun Culture & Working Environment", "Upskilling and Personal Growth", "Work with U.S. Counterparts", "Our Core Values:", "Trustworthy", "Reliable", "Driven", "Positive", "Kind"]
            },
            "industry": "Information & Communication Technology",
            "size": {
              "__typename": "CompanySize",
              "description": "101-1,000 employees"
            },
            "website": {
              "__typename": "Website",
              "url": "https:\\u002F\\u002Fwww.codev.com\\u002Fcareers\\u002F"
            }
          },
          "reviewsSummary": {
            "__typename": "ReviewsSummary",
            "overallRating": {
              "__typename": "OverallRating",
              "numberOfReviews": {
                "__typename": "NumberOfReviews",
                "value": 16
              },
              "value": 4.1875
            }
          },
          "perksAndBenefits": [{
            "__typename": "PerkAndBenefit",
            "title": "Life Insurance, Health\\u002FHMO Insurance +2 Dependents"
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Miscellaneous allowance"
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Annual Salary Increase "
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Remote Work Flexibility"
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Company Employee Morale Activities"
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Upskill \\u002F Online Training"
          }, {
            "__typename": "PerkAndBenefit",
            "title": "Company Issued Machine"
          }]
        },
        "ROOT_QUERY": {
          "__typename": "Query",
          "jobDetails:{\\"
          id\\ ":\\"
          82411354\\ "}": {
            "__typename": "JobDetails",
            "job": {
              "__typename": "Job",
              "sourceZone": "asia-6",
              "tracking": {
                "__typename": "JobTracking",
                "adProductType": null,
                "classificationInfo": {
                  "__typename": "JobTrackingClassificationInfo",
                  "classificationId": "6281",
                  "classification": "Information & Communication Technology",
                  "subClassificationId": "6287",
                  "subClassification": "Developers\\u002FProgrammers"
                },
                "hasRoleRequirements": true,
                "isPrivateAdvertiser": false,
                "locationInfo": {
                  "__typename": "JobTrackingLocationInfo",
                  "area": null,
                  "location": "Manila City",
                  "locationIds": ["2061106"]
                },
                "workTypeIds": "242",
                "postedTime": "3d ago"
              },
              "id": "82411354",
              "title": "Mid-Level Frontend Developer",
              "phoneNumber": null,
              "isExpired": false,
              "expiresAt": {
                "__typename": "SeekDateTime",
                "dateTimeUtc": "2025-03-29T12:59:59.999Z"
              },
              "isLinkOut": false,
              "contactMatches": [],
              "isVerified": true,
              "abstract": "Will work on a web-based application that allows users to input metropolitan areas and healthcare billing codes",
              "content({\\"
              platform\\ ":\\"
              WEB\\ "})": "\\u003Cp\\u003EThe Front-End Developer will work on the \\u003Cstrong\\u003Ehealthcare price benchmarking tool\\u003C\\u002Fstrong\\u003E, a web-based application that allows users to input metropolitan areas and healthcare billing codes (CPT codes) to receive relevant benchmark pricing data. This tool will include:\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EA \\u003Cstrong\\u003Esearchable interface\\u003C\\u002Fstrong\\u003E for users to input and retrieve pricing data\\u003C\\u002Fli\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003EData visualization elements\\u003C\\u002Fstrong\\u003E such as percentile ranges, regional heat maps, and interactive graphs\\u003C\\u002Fli\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003EExport functionality\\u003C\\u002Fstrong\\u003E for reports and datasets\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003EThe back-end, including data scrubbing and storage, will be handled by Simple Healthcare, with the front-end focused on presenting and visualizing this information.\\u003C\\u002Fp\\u003E\\u003Cp\\u003EThis is the first step in a broader roadmap, with a more \\u003Cstrong\\u003Ecomplex BI (Business Intelligence) dashboard\\u003C\\u002Fstrong\\u003E planned for future development in 2026.\\u003C\\u002Fp\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EResponsibilities\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EDevelop a \\u003Cstrong\\u003Escalable, user-friendly front-end\\u003C\\u002Fstrong\\u003E for the price benchmarking tool\\u003C\\u002Fli\\u003E\\u003Cli\\u003EImplement \\u003Cstrong\\u003Esearch and filtering functionalities\\u003C\\u002Fstrong\\u003E for healthcare pricing data\\u003C\\u002Fli\\u003E\\u003Cli\\u003EDevelop \\u003Cstrong\\u003Edata visualization elements\\u003C\\u002Fstrong\\u003E (heat maps, percentile charts, interactive tables)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EOptimize front-end performance for \\u003Cstrong\\u003Espeed and accessibility\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003ECollaborate with a \\u003Cstrong\\u003EUI\\u002FUX designer\\u003C\\u002Fstrong\\u003E to implement user-friendly designs\\u003C\\u002Fli\\u003E\\u003Cli\\u003EEnsure the front-end integrates seamlessly with the back-end APIs\\u003C\\u002Fli\\u003E\\u003Cli\\u003EMaintain \\u003Cstrong\\u003Eclean, scalable, and well-documented\\u003C\\u002Fstrong\\u003E code for future product expansion\\u003C\\u002Fli\\u003E\\u003Cli\\u003EWork closely with stakeholders to gather and refine requirements\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EQualifications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EMust-Have:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003E\\u003Cstrong\\u003E4+ years\\u003C\\u002Fstrong\\u003E of front-end development experience\\u003C\\u002Fli\\u003E\\u003Cli\\u003EStrong proficiency in \\u003Cstrong\\u003EReact.js\\u003C\\u002Fstrong\\u003E (preferred) or \\u003Cstrong\\u003EVue.js\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003ENext.js\\u003C\\u002Fstrong\\u003E (or similar SSR frameworks)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EStrong knowledge of \\u003Cstrong\\u003EJavaScript, TypeScript, HTML5, CSS3\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003ERESTful APIs and JSON data handling\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EFamiliarity with \\u003Cstrong\\u003Edata visualization libraries\\u003C\\u002Fstrong\\u003E (D3.js, Chart.js, Highcharts, or similar)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EKnowledge of \\u003Cstrong\\u003Estate management\\u003C\\u002Fstrong\\u003E (Redux, Context API)\\u003C\\u002Fli\\u003E\\u003Cli\\u003EUnderstanding of \\u003Cstrong\\u003Eresponsive and accessible web design\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003ENice-to-Have:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003EBI tools\\u003C\\u002Fstrong\\u003E or \\u003Cstrong\\u003Edata-heavy applications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EFamiliarity with \\u003Cstrong\\u003EMaterial UI, Tailwind CSS, or other UI libraries\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExperience with \\u003Cstrong\\u003Ehealthcare or financial tech applications\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EBasic understanding of \\u003Cstrong\\u003Ebackend technologies (Node.js, Express.js, PostgreSQL\\u002FMongoDB)\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fli\\u003E\\u003Cli\\u003EExposure to \\u003Cstrong\\u003EReact Native\\u003C\\u002Fstrong\\u003E for potential future mobile development\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E\\u003Cp\\u003E\\u003Cstrong\\u003EAdditional Skill\\u002F Ideal:\\u003C\\u002Fstrong\\u003E\\u003C\\u002Fp\\u003E\\u003Cul\\u003E\\u003Cli\\u003EBasic knowledge of \\u003Cstrong\\u003ESQL databases\\u003C\\u002Fstrong\\u003E.\\u003C\\u002Fli\\u003E\\u003Cli\\u003EHas understanding of \\u003Cstrong\\u003EUI\\u002FUX principles\\u003C\\u002Fstrong\\u003E and front-end performance optimization.\\u003C\\u002Fli\\u003E\\u003C\\u002Ful\\u003E",
              "status": "Active",
              "listedAt": {
                "__typename": "SeekDateTime",
                "label({\\"
                context\\ ":\\"
                JOB_POSTED\\ ",\\"
                length\\ ":\\"
                SHORT\\ ",\\"
                locale\\ ":\\"
                en - PH\\ ",\\"
                timezone\\ ":\\"
                UTC\\ "})": "3d ago",
                "dateTimeUtc": "2025-02-27T08:15:14.991Z"
              },
              "salary": {
                "__typename": "JobSalary",
                "currencyLabel({\\"
                zone\\ ":\\"
                asia - 6\\ "})": null,
                "label": "₱45,000 – ₱55,000 per month"
              },
              "shareLink({\\"
              locale\\ ":\\"
              en - PH\\ ",\\"
              platform\\ ":\\"
              WEB\\ ",\\"
              zone\\ ":\\"
              asia - 6\\ "})": "https:\\u002F\\u002Fph.jobstreet.com\\u002Fjob\\u002F82411354?tracking=SHR-WEB-SharedJob-asia-6",
              "workTypes": {
                "__typename": "JobWorkTypes",
                "label({\\"
                locale\\ ":\\"
                en - PH\\ "})": "Full time"
              },
              "advertiser": {
                "__ref": "Advertiser:60276460"
              },
              "location": {
                "__typename": "LocationInfo",
                "label({\\"
                locale\\ ":\\"
                en - PH\\ ",\\"
                type\\ ":\\"
                LONG\\ "})": "Manila City, Metro Manila"
              },
              "classifications": [{
                "__typename": "ClassificationInfo",
                "label({\\"
                languageCode\\ ":\\"
                en\\ "})": "Developers\\u002FProgrammers (Information & Communication Technology)"
              }],
              "products": {
                "__typename": "JobProducts",
                "branding": {
                  "__ref": "JobProductBranding:9d1d0514-d034-4e58-87c9-bde0b8073d60.1"
                },
                "bullets": ["Urgent hiring", "Work from home", "Great benefits with free online courses, HMO benefit and more"],
                "questionnaire": {
                  "__typename": "JobQuestionnaire",
                  "questions": ["What\'s your expected monthly basic salary?", "Which of the following types of qualifications do you have?", "How many years\' experience do you have as a frontend software developer?", "Which of the following programming languages are you experienced in?", "How many years of front end development experience do you have?", "Which of the following front end development libraries and frameworks are you proficient in?", "How would you rate your English language skills?"]
                },
                "video": {
                  "__typename": "VideoProduct",
                  "url": "https:\\u002F\\u002Fwww.youtube.com\\u002Fembed\\u002FNFJmx-M_M4U?t=70s",
                  "position": "BOTTOM"
                },
                "displayTags": []
              }
            },
            "companyProfile({\\"
            zone\\ ":\\"
            asia - 6\\ "})": {
              "__ref": "CompanyProfile:168553861426412"
            },
            "companySearchUrl({\\"
            languageCode\\ ":\\"
            en\\ ",\\"
            zone\\ ":\\"
            asia - 6\\ "})": "https:\\u002F\\u002Fph.jobstreet.com\\u002FCoDev-jobs\\u002Fat-this-company",
            "companyTags": [],
            "restrictedApplication({\\"
            countryCode\\ ":\\"
            PH\\ "})": {
              "__typename": "JobDetailsRestrictedApplication",
              "label({\\"
              locale\\ ":\\"
              en - PH\\ "})": null
            },
            "sourcr": null,
            "learningInsights({\\"
            locale\\ ":\\"
            en - PH\\ ",\\"
            platform\\ ":\\"
            WEB\\ ",\\"
            zone\\ ":\\"
            asia - 6\\ "})": {
              "__typename": "LearningInsights",
              "analytics": {
                "title": "Mid-Level Frontend Developer",
                "landingPage": "CA Role Salary",
                "resultType": "CareerAdviceSalaryTeaserSearch:frontend-developer",
                "entity": "career",
                "encoded": "title:Mid-Level Frontend Developer;landingPage:CA Role Salary;resultType:CareerAdviceSalaryTeaserSearch:frontend-developer;entity:career"
              },
              "content": "\\u003Cstyle\\u003E\\n  \\u002F* capsize font, don\'t change *\\u002F\\n  .capsize-heading4 {\\n    font-size: 20px;\\n    line-height: 24.66px;\\n    display: block;\\n    font-weight: 500;\\n  }\\n\\n  .capsize-heading4::before {\\n    content: \'\';\\n    margin-bottom: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-heading4::after {\\n    content: \'\';\\n    margin-top: -0.225em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText {\\n    font-size: 16px;\\n    line-height: 24.528px;\\n    display: block;\\n  }\\n\\n  .capsize-standardText::before {\\n    content: \'\';\\n    margin-bottom: -0.375em;\\n    display: table;\\n  }\\n\\n  .capsize-standardText::after {\\n    content: \'\';\\n    margin-top: -0.375em;\\n    display: table;\\n  }\\n\\n  @media only screen and (min-width: 740px) {\\n    .capsize-heading3 {\\n      font-size: 24px;\\n      line-height: 29.792px;\\n    }\\n\\n    .capsize-heading3::before {\\n      content: \'\';\\n      margin-bottom: -0.2292em;\\n      display: table;\\n    }\\n\\n    .capsize-heading3::after {\\n      content: \'\';\\n      margin-top: -0.2292em;\\n      display: table;\\n    }\\n  }\\n  \\u002F* end of capsize *\\u002F\\n\\n  \\u002F* LMIS css start here*\\u002F\\n  .lmis-root {\\n    margin: -32px;\\n    padding: 32px;\\n    font-family: SeekSans, \'SeekSans Fallback\', Arial, sans-serif;\\n    background: #beeff3;\\n    border-radius: 16px;\\n    color: #2e3849;\\n  }\\n\\n  .lmis-title {\\n    margin-bottom: 8px;\\n  }\\n\\n  .lmis-cta {\\n    min-height: 48px;\\n    display: flex;\\n    align-items: center;\\n    color: #2e3849;\\n    text-decoration: none;\\n  }\\n\\n  .lmis-cta-text {\\n    margin-right: 4px;\\n    font-weight: 500;\\n  }\\n\\n  .lmis-teaser-image {\\n    max-width: 96px;\\n  }\\n\\n  @media only screen and (min-width: 992px) {\\n    .lmis-root {\\n      margin: -48px;\\n    }\\n\\n    .lmis-wrapper {\\n      display: flex;\\n      flex-direction: row-reverse;\\n      justify-content: space-between;\\n      align-items: center;\\n    }\\n  }\\n\\u003C\\u002Fstyle\\u003E\\n\\n\\u003Cdiv class=\\"
              lmis - root\\ "\\u003E\\n  \\u003Cdiv class=\\"
              lmis - wrapper\\ "\\u003E\\n    \\u003Cdiv class=\\"
              lmis - teaser - image\\ "\\u003E\\n      \\u003Cimg\\n        src=\\"
              https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Fgirl_comparing_salaries.svg\\ "\\n        alt=\\"
              salary teaser image\\ "\\n      \\u002F\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n    \\u003Cdiv class=\\"
              lmis - content\\ "\\u003E\\n      \\u003Cdiv class=\\"
              capsize - heading4 lmis - title\\ "\\u003EWhat can I earn as a Front End Developer\\u003C\\u002Fdiv\\u003E\\n      \\u003Ca\\n        class=\\"
              lmis - cta\\ "\\n        href=\\"
              https: \\u002F\ \u002Fph.jobstreet.com\ \u002Fcareer - advice\ \u002Frole\ \u002Ffrontend - developer\ \u002Fsalary ? campaigncode = lrn : skj: sklm: cg: jbd: alpha\\ "\\n        target=\\"
              _blank\\ "\\n      \\u003E\\n        \\u003Cspan class=\\"
              capsize - standardText lmis - cta - text\\ "\\u003ESee more detailed salary information\\u003C\\u002Fspan\\u003E\\n        \\u003Cimg\\n          src=\\"
              https: \\u002F\ \u002Fcdn.seeklearning.com.au\ \u002Fmedia\ \u002Fimages\ \u002Flmis\ \u002Farrow_right.svg\\ "\\n          alt=\\"
              salary teaser link arrow\\ "\\n        \\u002F\\u003E\\n      \\u003C\\u002Fa\\u003E\\n    \\u003C\\u002Fdiv\\u003E\\n  \\u003C\\u002Fdiv\\u003E\\n\\u003C\\u002Fdiv\\u003E\\n"
            },
            "gfjInfo": {
              "__typename": "GFJInfo",
              "location": {
                "__typename": "GFJLocation",
                "countryCode": "PH",
                "country({\\"
                locale\\ ":\\"
                en - PH\\ "})": "Philippines",
                "suburb({\\"
                locale\\ ":\\"
                en - PH\\ "})": null,
                "region({\\"
                locale\\ ":\\"
                en - PH\\ "})": null,
                "state({\\"
                locale\\ ":\\"
                en - PH\\ "})": "Metro Manila",
                "postcode": null
              },
              "workTypes": {
                "__typename": "GFJWorkTypes",
                "label": ["FULL_TIME"]
              }
            },
            "seoInfo": {
              "__typename": "SEOInfo",
              "normalisedRoleTitle": "Front End Developer",
              "workType": "242",
              "classification": ["6281"],
              "subClassification": ["6287"],
              "where({\\"
              zone\\ ":\\"
              asia - 6\\ "})": "Manila City Metro Manila",
              "broaderLocationName({\\"
              locale\\ ":\\"
              en - PH\\ "})": "Manila City",
              "normalisedOrganisationName": "CoDev"
            }
          }
        }
      };\
      n
    </script>\n <script id="__LOADABLE_REQUIRED_CHUNKS__" type="application/json">
      [572, 160, 451]
    </script>
    <script id="__LOADABLE_REQUIRED_CHUNKS___ext" type="application/json">
      {
        "namedChunks": ["en-PH-translations", "JobDetailsPage"]
      }
    </script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/runtime~main-660254c82c2e38069a7f.js"></script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/vendor.react-54281ae222eeaf705804.js"></script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/vendor.reactUtils-177a67e51afc6fdcbef7.js"></script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/vendor.seekUtils-8efdb5c4d62e847c0d0d.js"></script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/vendors-b854ede820c5a8f1f70b.js"></script>\n <script async data-chunk="main" src="/static/ca-search-ui/houston/main-21f045c70749f232339b.js"></script>\n <script async data-chunk="en-PH-translations" src="/static/ca-search-ui/houston/en-PH-translations-716586b58fc181361a5d.js"></script>\n <script async data-chunk="JobDetailsPage" src="/static/ca-search-ui/houston/JobDetailsPage-a161baa010801caf3141.js"></script>\n <script async src="https://tags.tiqcdn.com/utag/seek/candidate-main/prod/utag.js"></script>\n <script defer>
      \
      n(function(b, r, a, n, c, h, _, s, d, k) {
        if (!b[n] || !b[n]._q) {
          for (; s < _.length;) c(h, _[s++]);
          d = r.createElement(a);
          d.async = 1;
          d.src = "https://cdn.branch.io/branch-latest.min.js";
          k = r.getElementsByTagName(a)[0];
          k.parentNode.insertBefore(d, k);
          b[n] = h
        }
      })(window, document, "script", "branch", function(b, r) {
        b[r] = function() {
          b._q.push([r, arguments])
        }
      }, {
        _q: [],
        _v: 1
      }, "addListener applyCode autoAppIndex banner closeBanner closeJourney creditHistory credits data deepview deepviewCta first getCode init link logout redeem referrals removeListener sendSMS setBranchViewData setIdentity track validateCode trackCommerceEvent logEvent disableTracking qrCode setRequestMetaData setAPIUrl getAPIUrl setDMAParamsForEEA".split(" "), 0);
    </script>\n \n \n
  </body>
</html>
""".strip()

if __name__ == '__main__':
    md_result = scrape_markdown(html_str)
    copy_to_clipboard(md_result['headings'][0]['content'])
    logger.success(md_result['headings'][0]['content'])
